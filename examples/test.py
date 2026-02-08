import os, sys, site

print("exe:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("no_site:", sys.flags.no_site, "isolated:", sys.flags.isolated)
print("getsitepackages:", site.getsitepackages())
print("sys.path contains site-packages?", any("site-packages" in p for p in sys.path))
print("---- sys.path ----")
for p in sys.path:
    print(p)
    if "site-packages" in p:
        print("site-packages found in", p)
        dir_contents = os.listdir(p)
        print("dir_contents:", dir_contents)
        for d in dir_contents:
            if "robot_skills" in d:
                print("robot_skills found in", d)
