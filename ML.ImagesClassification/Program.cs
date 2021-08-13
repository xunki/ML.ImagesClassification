using System;

namespace ML.ImagesClassification
{
    internal class Program
    {
        private static void Main()
        {
            Console.WriteLine("============ V1 ============");
            ImagesClassificationV1.Execute();

            Console.WriteLine();
            
            Console.WriteLine("============ V2 ============");
            ImagesClassificationV2.Execute();
        }
    }
}