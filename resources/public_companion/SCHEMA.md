# Synthetic household and scenario schemas

Persona JSONL records use ordinary `new-persona-000000000` identifiers; a new record has fourteen top-level fields: ID, record type, construction version, household size, currency, money unit, monthly income, essential spending, adjustable spending, opening cash, opening asset, opening debt, eligible task IDs and knowledge context. Financial amounts are nonnegative integer CNY cents. The opening-debt object contains principal, nominal annual basis points, term months and a quoted monthly payment. Household size is an authored 1–5-person setting.

`knowledge_context.geography` contains actual Wikidata QID and bilingual labels with compatible administrative types and parent. Its assignment type is `assigned_synthetic_context`, not an observed residence. Occupation is null; no occupation, education or preference is inferred from Wikidata geography.

The expectation-scale units are annual CNY per person. Generation applies monthly scale and an assigned household size. Real-survey aggregate means calibrate those scales; RNG shapes, household sizes, cash/asset/debt multiples and contracts are explicit design parameters. They do not identify a national joint distribution.

Scenario JSONL records retain the complete `household` with its ordinary ID and add a task, paired condition, assigned/effective state, English instruction/question, three complete options A/B/C and execution rules. Contract quotations are arithmetic terms, not realized payments. There are no expected answers, model choices, simulated outcome labels or executed monthly ledgers.

The completed release has 2,000,000 personas, 32,000 scenario inputs from 2,000 personas, and a first-20/320 example subset. Exact shard row counts and compressed bytes appear in `generation_info.json`. Full expansion of 2M households into eight tasks and two conditions would be 32M requests; that expansion is not materialized by this package.
