﻿using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML.ImagesClassification
{
    internal class Program
    {
        private static readonly string AssetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        private static readonly string ImagesFolder = Path.Combine(AssetsPath, "fish-images");
        private static readonly string TestImagesFolder = Path.Combine(AssetsPath, "test-images");

        private static readonly string InceptionTensorFlowModel =
            Path.Combine(AssetsPath, "inception", "tensorflow_inception_graph.pb");

        private static void Main()
        {
            // 训练模型
            var mlContext = new MLContext();
            var model = GenerateModel(mlContext);
            // 图片分类测试
            ClassifyImage(mlContext, model);
        }

        // Build and train model
        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages("input",
                    ImagesFolder, nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages("input",
                    InceptionSettings.IMAGE_WIDTH, InceptionSettings.IMAGE_HEIGHT, "input"))
                .Append(mlContext.Transforms.ExtractPixels("input",
                    interleavePixelColors: InceptionSettings.CHANNELS_LAST, offsetImage: InceptionSettings.MEAN))
                .Append(mlContext.Model
                    .LoadTensorFlowModel(InceptionTensorFlowModel)
                    .ScoreTensorFlowModel(new[] {"softmax2_pre_activation"}, new[] {"input"}, true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label"))
                .Append(mlContext.MulticlassClassification.Trainers
                    .LbfgsMaximumEntropy("LabelKey", "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            var images = Directory.GetFiles(ImagesFolder, "*", SearchOption.AllDirectories)
                .Select(file =>
                {
                    var path = file.Substring(ImagesFolder.Length).TrimStart('\\');
                    var label = path.Split('\\')[0];
                    return new ImageData
                    {
                        ImagePath = path,
                        Label = label
                    };
                }).ToList();

            var trainingData = mlContext.Data.LoadFromEnumerable(images);
            var model = pipeline.Fit(trainingData);
            return model;
        }

        public static void ClassifyImage(MLContext mlContext, ITransformer model)
        {
            var files = Directory.GetFiles(TestImagesFolder);
            foreach (var file in files)
            {
                var imageData = new ImageData
                {
                    ImagePath = file
                };
                var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
                var prediction = predictor.Predict(imageData);

                Console.WriteLine(
                    $"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        private struct InceptionSettings
        {
            public const int IMAGE_HEIGHT = 224;
            public const int IMAGE_WIDTH = 224;
            public const float MEAN = 117;
            public const float SCALE = 1;
            public const bool CHANNELS_LAST = true;
        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath { get; set; }

            [LoadColumn(1)]
            public string Label { get; set; }
        }

        public class ImagePrediction : ImageData
        {
            public float[] Score { get; set; }
            public string PredictedLabelValue { get; set; }
        }
    }
}