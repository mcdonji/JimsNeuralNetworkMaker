namespace Emidium.Jnnm.Console
{
    using System;
    using System.Collections.Generic;

    using Emidium.Jnnm.Core;

    class Program
    {
        static void Main(string[] args)
        {
            var input = new double[] { 0 };
            var desired = new double[] { 0.1 };
            System.Console.WriteLine("Training on [" + string.Join(",", input) + "] with desired of [" + string.Join(",", desired) + "]");


            int[] layerSizes = new[] { 1, 2, 1 };
            Emidium.Jnnm.Console.TransferFunction[] tFuncs = new TransferFunction[3]
            {
                Emidium.Jnnm.Console.TransferFunction.None,
                Emidium.Jnnm.Console.TransferFunction.Sigmoid,
                Emidium.Jnnm.Console.TransferFunction.Sigmoid
            };

            var bpn = new BackPropagationNetwork(layerSizes, tFuncs);
            var n00to10w = bpn.weight[0][0][0];
            var n00to11w = bpn.weight[0][0][1];
            var n10to20w = bpn.weight[1][0][0];
            var n11to20w = bpn.weight[1][1][0];
            var biasN10 = bpn.bias[0][0];
            var biasN11 = bpn.bias[0][1];
            var biasN20 = bpn.bias[1][0];
            double[] output =  new double[1];
            bpn.Run(ref input, out output);
            System.Console.WriteLine("Output before training of [" + string.Join(",", output) + "]");
            double error = 0;
            for (int i = 0; i < 10000; i++)
            {
                error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                if (i % 100 == 0)
                {
                    bpn.Run(ref input, out output);
                    System.Console.WriteLine("Result of [" + string.Join(",", output) + "] with error of [" + error + "]");
                }
            }

            System.Console.WriteLine(bpn.delta[0][0]);
            System.Console.WriteLine(bpn.delta[0][1]);
            System.Console.WriteLine(bpn.delta[1][0]);
            System.Console.WriteLine(bpn.weight[0][0][0]);
            System.Console.WriteLine(bpn.weight[0][0][1]);
            System.Console.WriteLine(bpn.weight[1][0][0]);
            System.Console.WriteLine(bpn.weight[1][1][0]);
            System.Console.WriteLine(bpn.bias[0][0]);
            System.Console.WriteLine(bpn.bias[0][1]);
            System.Console.WriteLine(bpn.bias[1][0]);



            var spec = new NeuralNetworkSpecification(new List<int> { 1, 2, 1 });
            var nn = new NeuralNetwork(spec);
            nn.TrainingRate = 0.15;
            nn.Momentum = 0.1;
            nn.Layers[0].Nodes[0].RightConnections[0].Weight = n00to10w;
            nn.Layers[0].Nodes[0].RightConnections[1].Weight = n00to11w;
            nn.Layers[1].Nodes[0].RightConnections[0].Weight = n10to20w;
            nn.Layers[1].Nodes[1].RightConnections[0].Weight = n11to20w;
            nn.Layers[1].Nodes[0].Bias = biasN10;
            nn.Layers[1].Nodes[1].Bias = biasN11;
            nn.Layers[2].Nodes[0].Bias = biasN20;

            //            for (int learningCount = 0; learningCount < 1000; learningCount++)
            //            {
            nn.Run(input);
            System.Console.WriteLine("Output before training of [" + string.Join(",", nn.Results()) + "] with error of [" + nn.CalcError(nn.Results(), desired) + "]");
            for (int i = 0; i < 10000; i++)
            {
                nn.Train(input, desired);
                if (i % 100 == 0)
                {
                    System.Console.WriteLine("Result of [" + string.Join(",", nn.Results()) + "] with error of [" + nn.LastError + "]");
                }
            }

            System.Console.WriteLine(nn.ToString());
            System.Console.ReadLine();
        }
    }
}