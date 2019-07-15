# custom_neuropype
Custom Neuropype Nodes

This repository contains implementation of Neuropype nodes that are not available in the academic edition.
As nodes become available in the academic edition, they will be removed from here.

Include this as a submodule in your project, preferably as a child of the working directory,
then use the nodes as in the following example.

```Python
import custom_neuropype as cn

result = cn.ImportPTB(filename)
```