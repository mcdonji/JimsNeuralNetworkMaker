namespace Emidium.Jnnm.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class NeuralNetworkMaker
    {
        public double HiddenLayerNodeMax { get; set; }
        public double HiddenLayerMax { get; set; }
        public double TrainingRate { get; set; }
        public double Momentum { get; set; }
        public double DataPointCount { get; set; }
        public double LearningCount { get; set; }
        public double TestingCount { get; set; }

        public NeuralNetworkMaker()
        {
            this.HiddenLayerNodeMax = 10;
            this.HiddenLayerMax = 5;
            this.TrainingRate = 0.1;
            this.Momentum = 0.1;
            this.DataPointCount = 100;
            this.LearningCount = 100;
            this.TestingCount= 100;
        }

        public NeuralNetwork Poof(Func<double[], double[]> mathFunc, int inputCount, int outputCount)
        {
            var random = new Random();
            List<NeuralNetwork> candidates = new List<NeuralNetwork>();
            for (int layerCount = 1; layerCount <= this.HiddenLayerMax; layerCount++)
            {
                for (int nodeCount = 2; nodeCount <= this.HiddenLayerNodeMax; nodeCount++)
                {
                    var neuralNetwork = new NeuralNetwork(new NeuralNetworkSpecification(inputCount, layerCount, nodeCount, outputCount));
                    neuralNetwork.TrainingRate = this.TrainingRate;
                    neuralNetwork.Momentum = this.Momentum;
                    neuralNetwork.InitializeWeights(random);
                    neuralNetwork.InitializeBiases(random);
                    neuralNetwork.SetLinearOutput();
                    Train(mathFunc, inputCount, random, neuralNetwork);
                    candidates.Add(neuralNetwork);
                }
            }

            for (int i = 0; i < this.TestingCount; i++)
            {
                var input = GetInputs(inputCount, random);
                double[] desired = mathFunc(input);
                foreach (var nn in candidates)
                {
                    nn.Test(input, desired);
                }
            }

            return candidates.OrderBy(nn => nn.LastTestingError).FirstOrDefault();
        }

        private static void Train(Func<double[], double[]> mathFunc, int inputCount, Random random, NeuralNetwork neuralNetwork)
        {
            Console.WriteLine("Training NN " + neuralNetwork.Spec);
            for (int learningCount = 0; learningCount < 100; learningCount++)
            {
                for (int datapointCount = 0; datapointCount < 1000; datapointCount++)
                {
                    var input = GetInputs(inputCount, random);
                    double[] desired = mathFunc(input);
                    neuralNetwork.Train(input, desired);
                }
            }
        }

        private static double[] GetInputs(int inputCount, Random random)
        {
            var input = new double[inputCount];
            for (int inputNumber = 0; inputNumber < inputCount; inputNumber++)
            {
                input[inputNumber] = random.NextDouble();
            }
            return input;
        }
    }
}