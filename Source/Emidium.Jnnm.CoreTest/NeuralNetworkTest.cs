using System.Data;

namespace Emidium.Jnnm.CoreTest
{
    using System;
    using System.Collections.Generic;

    using Emidium.Jnnm.Core;

    using Xunit;

    public class NeuralNetworkTest
    {

        [Fact]
        public void StaticXORNeuralNetworkTest()
        {
            var xorNnSpec = new NeuralNetworkSpecification(new List<int>() {2, 2, 1});
            var xorNn = new NeuralNetwork(xorNnSpec);
            xorNn.TrainingRate = 0.15;
            xorNn.Momentum = 0.1;
            xorNn.Layers[1].Nodes[0].Bias = -10;
            xorNn.Layers[1].Nodes[0].LeftConnections[0].Weight = 20;
            xorNn.Layers[1].Nodes[0].LeftConnections[1].Weight = 20;
            xorNn.Layers[1].Nodes[1].Bias = 30;
            xorNn.Layers[1].Nodes[1].LeftConnections[0].Weight = -20;
            xorNn.Layers[1].Nodes[1].LeftConnections[1].Weight = -20;
            xorNn.Layers[2].Nodes[0].Bias = -30;
            xorNn.Layers[2].Nodes[0].LeftConnections[0].Weight = 20;
            xorNn.Layers[2].Nodes[0].LeftConnections[1].Weight = 20;
            Assert.Equal(0d, Math.Round(xorNn.Run(new double[2] {0,0})[0]));
            Assert.Equal(0d, Math.Round(xorNn.Run(new double[2] {1,1})[0]));
            Assert.Equal(1d, Math.Round(xorNn.Run(new double[2] {0,1})[0]));
            Assert.Equal(1d, Math.Round(xorNn.Run(new double[2] {1,0})[0]));
            
            Console.WriteLine(xorNn.ToString());
        }

        [Fact]
        public void TrainXORNeuralNetworkWeightsAndBiasesTest()
        {
            var random = new Random();
            var xorNnSpec = new NeuralNetworkSpecification(new List<int>() {2, 2, 1});
            var xorNn = new NeuralNetwork(xorNnSpec);
            xorNn.TrainingRate = 0.15;
            xorNn.Momentum = 0.1;
            xorNn.InitializeWeights(random);
            xorNn.InitializeBiases(random);
            List<Tuple<double[], double[]>> dataset = new List<Tuple<double[], double[]>>();
            dataset.Add(new Tuple<double[], double[]>(new double[] {0,0}, new double[] {0} ));
            dataset.Add(new Tuple<double[], double[]>(new double[] {0,0}, new double[] {0} ));
            dataset.Add(new Tuple<double[], double[]>(new double[] {0,1}, new double[] {1} ));
            dataset.Add(new Tuple<double[], double[]>(new double[] {1,0}, new double[] {1} ));

            var lastError = Double.MaxValue;
            for (int learningCount = 0; learningCount < 10000; learningCount++)
            {
                xorNn.Train(dataset);
                Assert.True(lastError > xorNn.ErrorTotal);
                lastError = xorNn.ErrorTotal;
            }

            
            Assert.Equal(0d, Math.Round(xorNn.Run(new double[2] {0,0})[0]));
            Assert.Equal(0d, Math.Round(xorNn.Run(new double[2] {1,1})[0]));
            Assert.Equal(1d, Math.Round(xorNn.Run(new double[2] {0,1})[0]));
            Assert.Equal(1d, Math.Round(xorNn.Run(new double[2] {1,0})[0]));
                        
            
        }
        
        
        [Fact]
        public void CreateDefinedNeuralNetworkTest()
        {
            var random = new Random();
            var spec = new NeuralNetworkSpecification(new List<int> { 1,2,1});
            var nn = new NeuralNetwork(spec);
            nn.TrainingRate = 0.15;
            nn.Momentum = 0.1;
            nn.InitializeWeights(random);
            nn.InitializeBiases(random);
            var input = new double[] { 5, 10 };
            var desired = new double[] { 0.1 };
            var lastError = Double.MaxValue;
            for (int learningCount = 0; learningCount < 1000; learningCount++)
            {
                nn.TrainSingle(input, desired);
                Assert.True(lastError > nn.LastError);
                lastError = nn.LastError;
            }
        }

        [Fact]
        public void CreateAdditionNeuralNetworkTest()
        {
            var random = new Random();
            var spec = new NeuralNetworkSpecification(new List<int> { 2, 5,10, 5, 1 });
            var nn = new NeuralNetwork(spec);
            nn.TrainingRate = 0.15;
            nn.Momentum = 0.1;
            nn.InitializeWeights(random);
            nn.InitializeBiases(random);
            nn.SetLinearOutput();
            for (int learningCount = 0; learningCount < 1000000; learningCount++)
            {
                var first = random.Next(10);
                var second = random.Next(10);
                var input = new double[] { first, second };
                var desired = new double[] { first + second };
                nn.TrainSingle(input, desired);
            }
            var testinput = new double[] { random.Next(10), random.Next(10) };
            var actual = nn.Run(testinput)[0];
            Assert.Equal(testinput[0] + testinput[1], actual);
        }


        [Fact]
        public void CheckGradientApproximationOfBackPropagationTest()
        {
            var spec = new NeuralNetworkSpecification(new List<int> { 5, 1 });
            var nn = new NeuralNetwork(spec);
            nn.TrainingRate = 0.05;
            nn.Momentum = 0.01;
            var input = new double[] { 5, 10 };
            var desired = new double[] { 1,2,3,4,5 };
            nn.TrainSingle(input, desired);
        }

        [Fact]
        public void CreateAndNeuralNetworkTest()
        {
            var spec= new NeuralNetworkSpecification(new List<int>() { 2, 1 });
            var nn = new NeuralNetwork(spec);
            nn.TrainingRate = 0;
            nn.Momentum = 0;
            nn.Layers[1].Nodes[0].Bias = -30;
            nn.Layers[1].Nodes[0].TransferFunction= TransferFunctions.Sigmoid;
            nn.Layers[1].Nodes[0].TransferFunctionDerivitive= TransferFunctions.SigmoidDerivitive;
            var connections = nn.Layers[1].Nodes[0].LeftConnections;
            connections[0].Weight = 20;
            connections[1].Weight = 20;

            var at00 = (double)nn.Run(new double[] { 0, 0 })[0];
            Assert.True(Math.Abs(at00) < 0.0001 , "Expected value of about zero but got " + at00);
            Assert.True(Math.Abs((double)nn.Run(new double[] { 1, 0 })[0]) < 0.0001);
            Assert.True(Math.Abs(nn.Run(new double[] { 0, 1 })[0]) < 0.0001);
            Assert.True(Math.Abs(1 - nn.Run(new double[] { 1, 1 })[0]) < 0.0001);
        }
    }
}