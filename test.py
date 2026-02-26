from pxr import Usd

stage = Usd.Stage.Open("kinova_tasks/assets/robots/kinova/kinova_gen3_2f85.usd")


print("Root layer:", stage.GetRootLayer().identifier)
print("Sublayers:", stage.GetRootLayer().subLayerPaths)
print("Used layers:")
for layer in stage.GetUsedLayers():
    print(" ", layer.identifier)

prim = stage.GetPrimAtPath("/gen3_2f85_instantiable")
print("Prim stack:", prim.GetPrimStack())
