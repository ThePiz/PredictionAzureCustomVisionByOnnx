using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using ONNXClassifier.datastructures;

namespace ONNXClassifier
{
    class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
        }

        public struct TinyYoloModelSettings
        {
            // for checking Tiny yolo2 Model input and  output  parameter names,
            // you can use tools like Netron, 
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string ModelInput = "data";

            // output tensor name
            public const string ModelOutput = "classLabel";
            public const string Loss = "loss";


        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "data", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "data", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "data"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "data"))
                            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput, TinyYoloModelSettings.Loss }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<ResultTransformed> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            IDataView scoredData = model.Transform(testData);

            //retrieve single column
            //IEnumerable<string[]> probabilities = scoredData.GetColumn<string[]>(TinyYoloModelSettings.ModelOutput);
            //var scores = scoredData.GetColumn<IEnumerable<IDictionary<string, float>>>(TinyYoloModelSettings.Loss);

            var transformedDataPoints = mlContext.Data.CreateEnumerable<ResultTransformed>(scoredData, false).ToList();
          

            return transformedDataPoints;
        }

        public IEnumerable<ResultTransformed> Score(IDataView data)
        {
            Stopwatch stopwatch = Stopwatch.StartNew(); 
            var model = LoadModel(modelLocation);
            var result= PredictDataUsingModel(data, model);
            stopwatch.Stop();
            Console.WriteLine("Execution time in "+stopwatch.ElapsedMilliseconds + " ms");
            return result;
        }

        public class ResultTransformed
        {
            public string[] classLabel { get; set; }

            [OnnxSequenceType(typeof(IDictionary<string, float>))]
            public IEnumerable<IDictionary<string, float>> loss { get; set; }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("Suggested Prediction: "+ string.Join("\n", classLabel));
                sb.AppendLine();
                foreach (var tmp in loss)
                    foreach (var pair in tmp)
                    {
                        sb.Append($"Document type {pair.Key}: Reliability= {pair.Value:0.00} ");
                        sb.AppendLine();
                    }

                return sb.ToString();
            }
        }
    }
}