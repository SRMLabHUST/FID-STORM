import torch

## parameters
modelName   = r"D:\project\Pro7-mEDSR-STORM\experiment\exp6\ours\deconvolution\best.pkl"
shape       = 64

x = torch.randn((1,1,shape,shape),requires_grad=True)
y = torch.randn((1,1,shape,shape),requires_grad=True)
x = x.cuda()
y = y.cuda()

model = torch.load(modelName)
model.eval()

input_name  = "input"
output_name = "output"

## export dynamic input model
torch.onnx.export(model,                # model being run
                  x,                    # model input
                  "modelDynamic_ours_%dx%d.onnx"%(shape,shape),     # where to save the model (can be a file or file-like object)
                  opset_version=11,                                 # the ONNX version to export the model to
                  input_names=['input'],                            # the model's input names
                  output_names=['output'],                          # the model's output names
                  dynamic_axes={
                      input_name: {
                          0: 'batch_size'},
                      output_name:{
                          0: 'batch_size'}}
                  )

print("pytorchToOnnx finished!")

