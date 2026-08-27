"""The Bedrock harness (P13, wave2-start-plan.md) -- client, response cache,
usage meter, batch runner, router client, and the pinned price table.
Nothing in this package is imported by any CPU-only Tier A/B/C method;
``boto3`` is only required when ``llm_probability`` (src/methods/llm.py) is
actually used (see :mod:`src.bedrock.client`'s lazy import), per CLAUDE.md's
golden rule 5.
"""
