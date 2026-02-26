from pxr import Usd, UsdUtils

stage = Usd.Stage.Open("kinova_tasks/assets/robots/gen3_2f85/assembled_gen3_2f85.usd")
UsdUtils.FlattenLayerStack(stage).Export("kinova_tasks/assets/robots/gen3_2f85/flat_assembled_gen3_2f85.usd")
