# custom_neuropype
Custom Neuropype Nodes

This repository contains implementation of Neuropype nodes that are not available in the academic edition.
As nodes become available in the academic edition, they will be removed from here.

DO NOT commit directly to master. Always uses pull requests. This is because this repository may at times
have copies of nodes for which the master is in the developer's version of NeuroPype but not yet
the academic. Any edits to such nodes should be brought to Chad's attention and the best way to do that
is to use a pull request.

Include this as a submodule in your project, preferably as a child of the working directory,
then use the nodes as in the following example.

```Python
import custom_neuropype as cn

result = cn.ImportPTB(filename)
```