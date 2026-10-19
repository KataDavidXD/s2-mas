# Independent synthetic household companion

Dataset resources: [S²-MAS dataset resources](https://huggingface.co/collections/ya9919/s-mas-dataset-resources).

This companion supplies 2,000,000 newly assigned financial household personas and 32,000 unlabeled scenario inputs from the first 2,000 households, eight authored task families and two conditions. A 20-household/320-input subset is included in the data archive. These resources are separate from the paper's 2M and 12M execution cohorts and from existing model training.

The complete companion is distributed separately through [Hugging Face](https://huggingface.co/datasets/ya9919/s2-mas-synthetic-household-companion), including the 40 persona shards and scenario inputs. This source directory contains the generation code, schema, component notices and generation metadata. Dataset files are kept out of Git history.

Read [the dataset description](DATASET_README.md), [schema](SCHEMA.md) and [attribution](NOTICE.md). The new persona/scenario records use [research-use terms](LICENSE-DATA.md); the Wikidata facts remain [CC0](knowledge/LICENSE-CC0.md), the authored task design remains CC BY 4.0, and Python code uses [Apache-2.0](LICENSE-CODE). The [standalone Wiki component](../wikidata/README.md) contains the actual 38 entities and 66 relations.

To construct an independent small version, run the following from this directory:

```sh
python code/build_release.py --count 20 --seed 2026101919 --output regenerated
python code/validate_release.py --data-dir regenerated
```

The output directory must be new. Construction uses Python's standard library and the supplied knowledge/design files. A real-survey aggregate reference sets income/consumption expectation scales; synthetic financial states are assigned by RNG, while Wiki context supplies labels and parent/type relations. It does not infer observed residence, occupation or behavior.
