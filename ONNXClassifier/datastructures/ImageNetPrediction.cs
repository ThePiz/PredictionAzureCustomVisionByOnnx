using Microsoft.ML.Data;

namespace ONNXClassifier.datastructures
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}