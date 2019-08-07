# mixup_caffe
#create by fancy
this is a mixup python layer in caffe

your python data layer should like this:
layer {
  name: "mixupLayer"
  type: "Python"
  top: "data"
  top: "label"
  python_param{
    module: "mixupPythonLayer"
    layer: "DataLayer"
    param_str:'{"source_dir": "/data_2/dataset/fancydata/train.txt", "batch_size": 2}'
  }
}


your loss layer should like this:
layer {
  type: 'Python'
  name: 'loss'
  top: 'loss'
  bottom: 'fc_light_chebiao_2'
  bottom: 'label'
  python_param {
    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    module: 'mixupLossPythonLayer'
    # the layer name -- the class name in the module
    layer: 'LossDataLayer'
    param_str:'{"batch_size": 2, "numclass": 2}'
  }
  # set loss weight so Caffe knows this is a loss layer.
  # since PythonLayer inherits directly from Layer, this isn't automatically
  # known to Caffe
  loss_weight: 1
}


