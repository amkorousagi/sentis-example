using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.Sentis;
using UnityEngine;

public class SentisTest : MonoBehaviour
{
    public ModelAsset ModelAsset;
    public Texture2D ExampleImage;
    private Tensor _inputTensor;
    private IWorker _engineCPU;
    private IWorker _engineGPU;
    private Stopwatch _stopwatch;
    private IEnumerator _enumCPU;
    private IEnumerator _enumGPU;


    // Start is called before the first frame update
    void Start()
    {
        var model = ModelLoader.Load(ModelAsset);
        _engineCPU = WorkerFactory.CreateWorker(BackendType.CPU, model);
        _engineGPU = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        _inputTensor = TextureConverter.ToTensor(ExampleImage, channels: 3);
        _stopwatch = new Stopwatch();
    }

    // Update is called once per frame
    void Update()
    {
        _stopwatch.Restart();
        bool hasMoreCPU = false;
        _enumCPU = _engineCPU.ExecuteLayerByLayer(_inputTensor);
        for (int i=0;i<100000;i++)
        {
            hasMoreCPU = _enumCPU.MoveNext();
            if (!hasMoreCPU) break;
        }
        _stopwatch.Stop();
        print("cpu inference : " + _stopwatch.ElapsedMilliseconds + "ms");
        _stopwatch.Restart();
        // _engineCPU.Execute(_inputTensor); // schedule execution
        var outputTensorCPU = _engineCPU.PeekOutput("scores") as TensorFloat;
        outputTensorCPU.ToReadOnlyArray();
        _stopwatch.Stop();
        print("cpu download time (1st output): " + _stopwatch.ElapsedMilliseconds + "ms");
        _stopwatch.Restart();
        var outputTensorCPU2 = _engineCPU.PeekOutput("boxes") as TensorFloat;
        outputTensorCPU2.ToReadOnlyArray();
        _stopwatch.Stop();
        print("cpu download time (2nd output): " + _stopwatch.ElapsedMilliseconds + "ms");
        
        
        _stopwatch.Restart();
        bool hasMoreGPU = false;
        _enumGPU = _engineGPU.ExecuteLayerByLayer(_inputTensor);
        for (int i=0;i<100000;i++)
        {
            hasMoreGPU = _enumGPU.MoveNext();
            if (!hasMoreGPU) break;
        }
        _stopwatch.Stop();
        print("gpu inference : " + _stopwatch.ElapsedMilliseconds + "ms");
        _stopwatch.Restart();
        // _engineGPU.Execute(_inputTensor); // schedule execution
        var outputTensorGPU = _engineGPU.PeekOutput("scores") as TensorFloat;
        outputTensorGPU.CompleteOperationsAndDownload();
        outputTensorGPU.ToReadOnlyArray();
        _stopwatch.Stop();
        print("gpu download time (1st output): " + _stopwatch.ElapsedMilliseconds + "ms");
        _stopwatch.Restart();
        var outputTensorGPU2 = _engineGPU.PeekOutput("boxes") as TensorFloat;
        outputTensorGPU2.CompleteOperationsAndDownload();
        outputTensorGPU2.ToReadOnlyArray();
        _stopwatch.Stop();
        print("gpu download time (2nd output): " + _stopwatch.ElapsedMilliseconds + "ms");


    }
}