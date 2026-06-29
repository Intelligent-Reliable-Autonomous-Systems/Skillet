# Repairing Conditional Effects

Whenever you want to create a new experiment dataset, run
```bash
AML_CONFIG=$(python -m conditional_repair.baselines.experiments --config skillet_tasks/blocksworld-pick-place/repair-magnet/repair-effects.json --create)
```

This will give you a config JSON file for that dataset instance. You can run multiple baselines on the same data. Or you can just use `AML_CONFIG=skillet_tasks/blocksworld-pick-place/repair-magnet/repair-effects.json` to get a new dataset each time.

Run 'orcam', 'olam', or 'random' online baselines:
```
python -m conditional_repair.baselines.online.online_agent --config $AML_CONFIG --approach orcam
```

Run 'orcam', 'predicators', 'conditional-sam' offline baselines with 'orcam' or 'random' data:
```
python -m conditional_repair.baselines.offline.offline_agent --config $AML_CONFIG --approach conditional-sam --data orcam
```
I wouldn't worry about the offline baselines, because they may require some configuration. Just collect the data.