# CV_Tasks
 
<h2>TASK-1: Conv2dGroupinglayer</h2>
<h3> Overview on Methodology</h3>
<ul>
<li>Input channels and the number of kernels for the Output channels both are divided into 16 groups</li>
<li>Conv2d layer is created for each of these 16 groups and stored using nn.ModuleList()</li>
<li>The given weights and bias are utilzed for the custom_grouped_conv2d layer-operation</li>
<li>These weights and bias are chunked into groups and copied to their respective layers</li>
<li>Each group has weight_dim:(8,4,3,3) and bias_dim:(8)</li>
<li>Conv2d operation is performed on all the 16 groups and the outputs are concatenated</li>
<li>Output_shape:(2,128,100,100)</li>
<li>torch.isclose() verifies that the output from the CustomGroupedConv2d is same as grouped_layer</li>
</ul>

<h3> References</h3>
<ul>
<li><a href='https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html'> nn.Conv2d </a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.chunk.html'>torch.chunk</a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.cat.html'>torch.cat </a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.isclose.html'>torch.isclose </a></li>
</ul>


<h2>TASK-2: ImageAugmentation</h2>
<h3> Overview on Methodology</h3>
<ul>
<li>cv2 reads image as nd-array [H,W,C] ,BGR color-code format and uint8 dtype</li>
<li>torchvision.Transforms takes input as PIL or Tensor[..,H,W]</li>
<li>Therefore, the image is converted to Tensor and permuted</li>
<li>The given set of augmentation is carried over in sequence with Compose class</li>
<li>To dispaly the output, the tensor is converted back to numpy and transposed (equivalent to tensor permute)</li>
</ul>

<h3> References</h3>
<ul>
<li><a href='https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html'> OpenCV </a></li>
<li><a href='https://pytorch.org/vision/stable/transforms.html'> Pytorch_Transformations </a></li>
</ul>
<h2>TASK-3</h2>
<h3> Overview on Methodology</h3>
<ul>
<li>The onnx file is loaded and opened</li>
<li>Model-Layer information has been translated to pytorch-framework</li>
<li>Weights and Bias has been initialised separately for conv-layers and linear-layers</li>
<li>Output is obtained from the model for the given input</li>
</ul>

<h3> References</h3>
<ul>
<li><a href='https://pytorch.org/docs/stable/onnx.html'> open onnx files in pytorch framework </a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html'> nn.Conv2d </a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d'> nn.BatchNorm2d </a></li>
<li><a href='https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html'> nn.Sigmoid </a></li>
<li><a href='https://pytorch.org/docs/stable/nn.init.html'> Initialize weights and bias in pytorch model</a></li>
<li><a href='https://pytorch.org/docs/stable/notes/modules.html?highlight=modules'> Accessing pytorch modules and parameters</a></li>
</ul>
