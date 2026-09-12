"""Сопоставление синхронного и асинхронного транспорта без генерации."""
import asyncio
import json
import time
import reasoning_round as rr

case=json.loads((rr.sdk.OLD/'cases/CI10071259.json').read_text())
judge,_=rr.make_judge(case,'pro','low','text')
client=judge.llm._client
secrets=[v for v in [judge.llm.credentials,judge.llm.access_token] if v]

def failure(error):
    chain=[]
    while error is not None:
        detail=str(error)
        for secret in secrets:
            detail=detail.replace(secret,'[REDACTED]')
        chain.append({'type':type(error).__name__,'detail':detail})
        error=error.__cause__
    return chain

async def main():
    results={'tls_max':client._settings.ssl_context.maximum_version.name,'token_seeded':bool(judge.llm.access_token),'probes':[]}
    for kind in ['sync','async']:
        started=time.monotonic()
        try:
            response=client.get_models() if kind=='sync' else await client.aget_models()
            result={'status':'ok','models':len(response.data)}
        except Exception as error:
            result={'status':'error','chain':failure(error)}
        result.update(kind=kind,seconds=round(time.monotonic()-started,3))
        results['probes'].append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    (rr.sdk.OUT/'transport-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

asyncio.run(main())
