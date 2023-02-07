# FID-STORM

FID-STORM is a real-time processing method for single molecule localization microscopy with deep learning, which is based on improved residual convolutional network. This method can achieve real-time processing of raw images up to 256×256 pixels @ Nvidia RTX 2080 Ti graphic card at a speed of 7.31 ms/frame, shorter than the typical exposure time of 10~30 ms. Moreover, compared with a popular interpolated image-based method called Deep-STORM, FID-STORM enables a speed gain of ~25 times, without loss of reconstruction accuracy.

The following picture shows the schematic comparison of proposed FID-STORM with the conventional Interpolation-based SMLM model.
![](https://github.com/SRMLabHUST/FID-STORM/blob/main/data/scheme.png)

For more details, please refer to ***User‘s Guide of FID-STORM.pdf*** and our paper ***Deep learning using residual deconvolutional network enables real-time high-density single-molecule localization microscopy***.

The repository consists of 3 folders, that are data, ImageJ plugin, and Source code.

| Folders       | Function                                                     |
| ------------- | ------------------------------------------------------------ |
| data          | Consists of  relative model input and output data.           |
| ImageJ plugin | A .jar file, which can be loaded by ImageJ as a plugin.      |
| Source code   | 1) The source code of FID-STORM with c++ and java in folder cplusplus and Java using for inferring; <br>2) The source code of FID-STORM with python in folder python using for training a model |

## License

There are two licenses for different part of the ANNA-PALM code: a [`MIT license`](https://github.com/imodpasteur/ANNA-PALM/blob/master/AnetLib/LICENSE) is applied to files inside the `AnetLib` folder. A [`Non-commercial License Agreement`](https://github.com/imodpasteur/ANNA-PALM/blob/master/license.pdf) is applied to all other files.

Declaration
This program is free software: you can redistribute it and/or modify it under the terms of the GNU LESSER GENERAL PUBLIC LICENSE as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU LESSER GENERAL PUBLIC LICENSE for more details.

You should have received a copy of the GNU LESSER GENERAL PUBLIC LICENSE along with this program. If not, see https://www.gnu.org/licenses/.

For more questions, please contact Prof. Zhengxia Wang at "zxiawang@hainanu.edu.cn" or author Zhiwei Zhou at "1258806185@qq.com".
