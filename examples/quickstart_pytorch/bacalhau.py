# from bacalhau_sdk.api import submit
# from bacalhau_sdk.config import get_client_id

# data = dict(
#     apiversion='V1beta1',
#     clientid=get_client_id(),
#     spec=dict(
#         engine="Docker",
#         verifier="Noop",
#         publisher="Estuary",
#         docker=dict(
#             image="ubuntu",
#             entrypoint=["sleep", "15"],
#         ),
#         deal=dict(concurrency=3, confidence=0, min_bids=0),
#         inputs=[          
#           dict(
#             storagesource="ipfs",
#             cid="QmWG3ZCXTbdMUh6GWq2Pb1n7MMNxPQFa9NMswdZXuVKFUX",
#             path="/datasets",
#           )
#         ],
#     ),
# )

# submit(data)

import pprint
import time

from bacalhau_sdk.api import submit
from bacalhau_sdk.config import get_client_id
from bacalhau_apiclient.models.storage_spec import StorageSpec
from bacalhau_apiclient.models.spec import Spec
from bacalhau_apiclient.models.job_spec_language import JobSpecLanguage
from bacalhau_apiclient.models.job_spec_docker import JobSpecDocker
from bacalhau_apiclient.models.deal import Deal
from bacalhau_apiclient.models.publisher_spec import PublisherSpec
from bacalhau_sdk.api import results

# TODO: IPFS Python SDK?

data = dict(
    APIVersion='V1beta1',
    ClientID=get_client_id(),
    Spec=Spec(
        engine="Docker",
        verifier="Noop",
        publisher_spec=PublisherSpec(type="ipfs"),
        docker=JobSpecDocker(
            image="ubuntu",
            entrypoint=["echo", "Hello, cruel World!"],
        ),
        language=JobSpecLanguage(job_context=None),
        wasm=None,
        resources=None,
        timeout=1800,
        outputs=[
            StorageSpec(
                storage_source="IPFS",
                name="outputs",
                path="/outputs",
            )
        ],
        deal=Deal(concurrency=1, confidence=0, min_bids=0),
        do_not_track=False,
    ),
)

job = submit(data)
job_id = job.job.metadata.id
pprint.pprint(job)
print(job_id)
res = results(job_id)
while not res.results:
    time.sleep(1)
    res = results(job_id)
    print(".", end="")
print("\n" + str(res))
print(f"https://ipfs.io/ipfs/{res.results[0].data.cid}/stdout")