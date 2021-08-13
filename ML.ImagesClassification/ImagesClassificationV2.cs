using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace ML.ImagesClassification
{
    public class ImagesClassificationV2
    {
        private static readonly string AssetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        private static readonly string ImagesFolder = Path.Combine(AssetsPath, "fish-images");
        private static readonly string TestImagesFolder = Path.Combine(AssetsPath, "test-images");
        private static readonly string ModelPath = Path.Combine(AssetsPath, "model_v2.zip");

        public static void Execute()
        {
            // 训练模型
            var mlContext = new MLContext();
            var model = GenerateModel(mlContext, true);

            // 图片分类测试
            ClassifyImage(mlContext, model);
        }

        private static ITransformer GenerateModel(MLContext mlContext, bool useCache)
        {
            if (useCache && File.Exists(ModelPath))
            {
                var m = mlContext.Model.Load(ModelPath, out _);
                return m;
            }

            var images = Directory.GetFiles(ImagesFolder, "*", SearchOption.AllDirectories)
                .Select(file =>
                {
                    var label = file[ImagesFolder.Length..].TrimStart('\\').Split('\\')[0];
                    return new ModelInput
                    {
                        Image = File.ReadAllBytes(file),
                        Label = label
                    };
                }).ToList();

            var imageData = mlContext.Data.LoadFromEnumerable(images);
            var shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey");

            var preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

            var trainSplit = mlContext.Data.TrainTestSplit(preProcessedData, 0.3);
            var trainSet = trainSplit.TrainSet;
            var validationSet = trainSplit.TestSet;

            var classifierOptions = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = Console.WriteLine,
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = trainingPipeline.Fit(trainSet);
            if (useCache)
                mlContext.Model.Save(model, imageData.Schema, ModelPath);
            return model;
        }

        private static void ClassifyImage(MLContext mlContext, ITransformer model)
        {
            var imageData = mlContext.Data.LoadFromEnumerable(Directory.GetFiles(TestImagesFolder).Select(file =>
                new ModelInput
                {
                    Image = File.ReadAllBytes(file),
                    Label = Path.GetFileName(file)
                }));

            var preprocessingPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey");

            var preProcessedData = preprocessingPipeline.Fit(imageData).Transform(imageData);
            var inputs = mlContext.Data.CreateEnumerable<ModelInput>(preProcessedData, true);

            var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            foreach (var prediction in inputs.Select(input => predictor.Predict(input)))
            {
                Console.WriteLine(
                    $"Image: {prediction.Label} | Predicted Value: {prediction.PredictedLabel} with score: {prediction.Score.Max()}");
            }
        }

        public class ModelInput
        {
            public string Label { get; set; }
            public byte[] Image { get; set; }
            public uint LabelAsKey { get; set; }
        }

        public class ModelOutput
        {
            public string Label { get; set; }
            public string PredictedLabel { get; set; }
            public float[] Score { get; set; }
        }
    }
}