"""Safe budgeted Responses calls. No credentials or image payloads in logs."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from ..credentials import openai_key

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'cache/contexts/api_budget.json'
MODEL = 'gpt-4.1-mini-2025-04-14'
INPUT_PRICE, OUTPUT_PRICE = .4, 1.6


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class LedgerLock:
    def __enter__(self):
        self.path = LEDGER.with_suffix('.mentor.lock')
        started = time.monotonic()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.monotonic() - started > 15:
                    raise RuntimeError('Budget ledger lock busy; do not issue a paid request')
                time.sleep(.05)

    def __exit__(self, *_):
        os.close(self.fd)
        self.path.unlink()


def reconcile_confirmed_receipts(out):
    """Release ONLY excess reservations of six documented completed E07 calls.

    Original failed/unknown reservations and old unitemized reservations remain.
    Measured cost is already in the ledger, so never add it a second time.
    """
    with LedgerLock():
        ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
        done = set(ledger.get('settled_receipt_sha256', []))
        reconciled = []
        confirmed=[]
        base = ROOT / 'runs/improvements_20261005'
        for path in sorted(base.rglob('receipt.json')):
            receipt = json.loads(path.read_text(encoding='utf-8'))
            sha = digest(path.read_bytes())
            if receipt.get('status') != 'completed':
                continue
            usage = receipt.get('usage', {})
            if not all(isinstance(usage.get(k), int) for k in ['input_tokens', 'output_tokens']):
                continue
            model = receipt.get('returned_model')
            rates = ledger['followup_model_prices_usd_per_million'].get(model)
            if not rates:
                continue
            cost = (usage['input_tokens'] * rates['input'] + usage['output_tokens'] * rates['output']) / 1e6
            reserve = receipt.get('reservation_usd')
            if reserve is None or abs(cost - receipt['estimated_cost_usd']) > 1e-10 or cost > reserve:
                raise ValueError('Receipt budget reconciliation failed; preserve the ledger')
            confirmed.append({'receipt':str(path.relative_to(ROOT)),'sha256':sha,
                              'old_reservation_usd':reserve,'confirmed_usage_estimate_usd':cost,
                              'already_settled':sha in done})
            if sha in done:continue
            ledger['reserved_usd'] -= reserve - cost
            done.add(sha)
            reconciled.append({'receipt':str(path.relative_to(ROOT)), 'sha256':sha,
                               'old_reservation_usd':reserve, 'confirmed_usage_estimate_usd':cost})
        ledger['settled_receipt_sha256'] = sorted(done)
        ledger['settlement_note'] = 'Completed response usage settles its own reservation only; failed/unresolved holds retained.'
        write(LEDGER, ledger)
        write(out, {'reconciled':reconciled,'confirmed_receipts':confirmed,
                    'remaining_conservative_usd':min(4., ledger['limit_usd'])-ledger['reserved_usd'],
                    'ledger_measured_estimate_usd':ledger['measured_estimate_usd'], 'invoice_authoritative':True,
                    'failed_reservations_released':False})


def reserve(amount, request_sha):
    with LedgerLock():
        ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
        if ledger['reserved_usd'] + amount > min(4., ledger['limit_usd']):
            raise RuntimeError('Approved cumulative $4 budget reached; no paid request issued')
        ledger['reserved_usd'] += amount
        ledger['attempts'] += 1
        ledger.setdefault('mentor_pending', {})[request_sha] = amount
        write(LEDGER, ledger)


def settle(amount, cost, request_sha):
    with LedgerLock():
        ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
        if cost > amount:
            # Retain actual spend, then stop; never conceal an unexpected bill.
            ledger['reserved_usd'] += cost - amount
            write(LEDGER, ledger)
            raise RuntimeError('Actual response usage exceeded its conservative reservation')
        ledger['reserved_usd'] -= amount - cost
        ledger['measured_estimate_usd'] += cost
        ledger['successful_calls'] += 1
        ledger.setdefault('mentor_pending', {}).pop(request_sha, None)
        write(LEDGER, ledger)


def request(content, schema, evidence, destination, paid=False, recover_transport=False):
    """Resume accepted responses; never automatically retry rejected/failed calls."""
    destination = Path(destination)
    payload = {'model':MODEL, 'input':[{'role':'user', 'content':content}], 'store':False,
               'temperature':0, 'max_output_tokens':600,
               'text':{'format':{'type':'json_schema', 'name':'visible_process_state', 'strict':True, 'schema':schema}}}
    sha = digest(payload)
    if (destination / 'accepted.json').exists():
        cached = json.loads((destination / 'accepted.json').read_text(encoding='utf-8'))
        if cached['request_sha256'] != sha:
            raise ValueError('Cached request differs; choose a new run, never overwrite accepted evidence')
        return cached['state']
    if (destination/'failure.json').exists() and recover_transport and not (destination/'receipt.json').exists():
        failure=json.loads((destination/'failure.json').read_text(encoding='utf-8'))
        recovery=destination/'transport_recovery_1'
        if failure.get('error')!='connection/timeout' or recovery.exists():
            raise RuntimeError('Only one explicit transport recovery is allowed; no response retry')
        # Old failed reservation remains. Use a separate directory/receipt for
        # the explicit environment repair, never silently overwrite failure.
        recovery.mkdir(parents=True)
        write(recovery/'recovery_policy.json',{'one_shot':True,'old_reservation_retained':True,
                                              'old_failure':failure,'timeout_seconds':100})
        return request(content,schema,evidence,recovery,paid=paid,recover_transport=False)
    if (destination / 'receipt.json').exists() or (destination / 'failure.json').exists():
        raise RuntimeError('Previous paid response/failure needs review; no automatic paid retry')
    if not paid:
        raise RuntimeError('Live VLM inference requires explicit --paid; cached-only mode cannot invent states')
    # UTF-8 byte count is a conservative text-token ceiling. Image count uses
    # official <=1536 32px patches, 1.62 multiplier at actual transmitted sizes.
    text_bytes = sum(len(x.get('text','').encode()) for x in content)
    text_bytes += len(json.dumps(schema).encode()) + 1024
    image_tokens = sum(math.ceil(math.ceil(e['width']/32)*math.ceil(e['height']/32)*1.62) for e in evidence)
    bound = (text_bytes + image_tokens) * INPUT_PRICE / 1e6 + 600 * OUTPUT_PRICE / 1e6
    # A recovery has the same payload hash, but is a DIFFERENT billed attempt.
    # Do not overwrite/remove an unresolved original attempt's ledger entry.
    attempt_key=sha+':'+digest(str(destination.resolve()))[:16]
    reserve(bound, attempt_key)
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / 'request_meta.json', {'request_sha256':sha, 'attempt_key':attempt_key,'model':MODEL, 'schema_sha256':digest(schema),
          'images':evidence, 'reservation_usd':bound, 'future_images_supplied':False, 'test_labels_supplied':False})
    req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(payload).encode(),
                                 headers={'Authorization':'Bearer '+openai_key(), 'Content-Type':'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=100) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        write(destination / 'failure.json', {'http_status':error.code, 'retry':False, 'reservation_retained':True})
        raise RuntimeError(f'VLM API HTTP {error.code}; no automatic retry') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        write(destination / 'failure.json', {'error':'connection/timeout', 'retry':False, 'reservation_retained':True})
        raise RuntimeError('VLM API connection/timeout; no automatic retry') from None
    usage = result.get('usage', {})
    if not all(isinstance(usage.get(k), int) for k in ['input_tokens', 'output_tokens']):
        write(destination / 'failure.json', {'error':'missing_usage', 'retry':False, 'reservation_retained':True})
        raise RuntimeError('Missing usage; cost reservation retained')
    cost = (usage['input_tokens']*INPUT_PRICE + usage['output_tokens']*OUTPUT_PRICE)/1e6
    receipt = {'request_sha256':sha, 'requested_model':MODEL, 'returned_model':result.get('model'),
               'response_id':result.get('id'), 'status':result.get('status'), 'usage':usage,
               'estimated_cost_usd':cost, 'reservation_usd':bound, 'wall_seconds':time.perf_counter()-started}
    write(destination / 'receipt.json', receipt)
    settle(bound, cost, attempt_key)
    text = ''.join(p.get('text','') for item in result.get('output',[]) for p in item.get('content',[])
                   if p.get('type') == 'output_text')
    write(destination / 'raw_output.json', {'text':text, 'status':result.get('status')})
    if result.get('status') != 'completed' or result.get('model') != MODEL:
        raise RuntimeError('Incomplete/different model response; not treated as a normal frame')
    try:
        state = json.loads(text)
    except ValueError:
        raise RuntimeError('VLM response invalid JSON; no surrogate state/no automatic retry') from None
    return state
