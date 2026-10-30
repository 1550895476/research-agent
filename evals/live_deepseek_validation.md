# Live DeepSeek + Tavily validation

Date: 2026-08-24

## Scope

The live evaluation used an OpenAI-compatible DeepSeek endpoint with model
`deepseek-v4-flash` and Tavily web search. Credentials were injected into the running
process only and are not stored in this repository.

Configuration used for the bounded evaluation:

- 2 initial searches per research task;
- 1 follow-up search per quality round;
- 4 maximum searches per task;
- quality threshold 70;
- at least 3 unique sources and 2 independent domains.

## Findings before improvements

- DeepSeek structured output worked for requirement parsing, planning, tool selection,
  research synthesis and task quality assessment.
- A three-module official-source comparison performed 12 successful Tavily calls and
  collected 54 evidence records.
- Tasks scored 35, 15 and 55. They correctly failed the quality gate because search
  returned community and third-party pages despite the official-only requirement.
- The Researcher batch exceeded the client's 300-second request timeout. Individual
  parallel Researcher spans lasted up to approximately 263 seconds.
- Model calls, rather than Tavily, dominated latency. Tavily calls generally completed
  in 3–5 seconds, while structured research and verification calls took approximately
  15–56 seconds each.

## Improvements implemented

1. `source_domains` was added to Planner and task schemas. The Planner converts
   source restrictions into Tavily domain allowlists for initial and follow-up searches.
2. `source_url_prefixes` was added for shared hosts such as GitHub. Search results are
   deterministically filtered after retrieval, so an official repository requirement
   does not admit unrelated repositories on the same domain.
3. Researcher and task-level Verifier evidence rendering now compacts and distributes
   context across every retrieved source instead of truncating later searches from one
   large JSON string.

## Validation after improvements

- The live Planner produced domain allowlists for LangGraph, CrewAI and AutoGen.
- The final live Planner smoke test produced exact prefixes including
  `https://github.com/langchain-ai/langgraph` and
  `https://langchain-ai.github.io/langgraph`.
- A focused LangGraph run passed domain restrictions to all four Tavily calls and
  collected 18 evidence records. Its score increased from the earlier LangGraph score
  of 35 to 42, but still failed the strict completion criteria, which is preferable to
  accepting unsupported work.
- A live broad GitHub query returned zero items after prefix filtering rather than
  leaking unrelated repositories. A precise site query returned two URLs, both under
  the required official repository prefix.
- The complete offline regression suite contains 63 passing tests; Ruff and JavaScript
  syntax checks also pass.

## Remaining limitations

- The final prefix-filtered policy was smoke-tested with live planning and live Tavily,
  but a complete end-to-end report was not rerun because the previous live attempts
  showed high model latency and repeated calls would consume additional quota.
- Domain and URL allowlists improve precision but can reduce recall. Planner-generated
  site-specific queries are important when prefix filtering is enabled.
- A single research segment can remain synchronous for several minutes. A production
  version should expose background run IDs or a streaming event endpoint and support
  fast, standard and deep research budgets.
- Provider-reported token usage and monetary cost are not yet captured; trace token
  values remain estimates.
