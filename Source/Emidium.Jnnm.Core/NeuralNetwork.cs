namespace Emidium.Jnnm.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class NeuralNetwork
    {
        public List<Layer> Layers = new List<Layer>();
        public NeuralNetworkSpecification Spec { get; set; }
        public double TrainingRate { get; set; }
        public double Momentum { get; set; }
        public double LastError { get; set; }
        public double TrainingCount { get; set; }
        public double ErrorTotal { get; set; }
        public double AverageError => this.ErrorTotal / this.TrainingCount;
        public double TestingCount { get; set; }
        public double TestingErrorTotal { get; set; }
        public double LastTestingError { get; set; }
        public double AverageTestingError => this.TestingErrorTotal / this.TestingCount;



        public NeuralNetwork(NeuralNetworkSpecification spec)
        {
            this.Spec = spec;
            for (int layerIndex = 0; layerIndex < this.Spec.NumberOfLayers; layerIndex++)
            {
                this.Layers.Add(new Layer(layerIndex, this.Spec.IsOutput(layerIndex), this.Spec.LayerNodes[layerIndex]));
            }

            for (int i = 0; i < this.Spec.NumberOfLayers-1; i++)
            {
                var thisLayer = this.Layers[i];
                var nextLayer = this.Layers[i +1];
                foreach (var thisLayerNode in thisLayer.Nodes)
                {
                    foreach (var nextLayerNode in nextLayer.Nodes)
                    {
                        var connection = new NodeConnection(thisLayerNode, nextLayerNode);
                        thisLayerNode.RightConnections.Add(connection);
                        nextLayerNode.LeftConnections.Add(connection);
                    }
                }
            }
        }

        public double[] Run(double[] input)
        {
            this.Layers[0].SetOutput(input);
            for (int i = 1; i < this.Layers.Count; i++)
            {
                this.Layers[i].ForwardPropagate();
            }
            return this.Results();
        }

        public void BackPropagate(double[] desired)
        {
            this.Layers[this.Layers.Count-1].SetDelta(desired);
            for (int i = this.Layers.Count-2; i >= 0; i--)
            {
                this.Layers[i].BackPropagateDeltas();
            }
            for (int i = 0; i <= this.Layers.Count-1; i++)
            {
                var layer = this.Layers[i];
                layer.SetWeights(this.TrainingRate, this.Momentum);
                layer.SetBias(this.TrainingRate, this.Momentum);
            }
        }

        public void Train(double[] input, double[] desired)
        {
            this.Run(input);
            this.BackPropagate(desired);
            this.Run(input);
            this.LastError = this.CalcError(this.Results(), desired);
            this.ErrorTotal += this.LastError;
            this.TrainingCount++;
        }

        public double CalcError(double[] results, double[] desired)
        {
            double error = 0;
            for (int i = 0; i < desired.Length; i++)
            {
                error += Math.Pow(desired[i] - results[i], 2);
            }
            return error;
        }

        public double[] Results()
        {
            return this.LastLayer().Nodes.Select(n => n.Output).ToArray();
        }

        private Layer LastLayer()
        {
            return this.Layers[this.Layers.Count-1];
        }

        public void InitializeWeights(Random random)
        {
            for (int i = 0; i < this.Layers.Count-1; i++)
            {
                foreach (var node in this.Layers[i].Nodes)
                {
                    node.RightConnections.ForEach(con =>con.Weight = random.NextDouble());
                }
            }
        }

        public void InitializeBiases(Random random)
        {
            foreach (var layer in this.Layers)
            {
                layer.Nodes.ForEach(node => node.Bias = random.NextDouble());
            }
        }

        public override string ToString()
        {
            string result = "NN " + Environment.NewLine;
            foreach (var layer in this.Layers)
            {
                result += " Layer : " +  layer.LayerIndex  + Environment.NewLine;

                foreach (var node in layer.Nodes)
                {
                    result += "  Node : " + node.Code + Environment.NewLine;
                    result += "      Input  : " + node.Input + Environment.NewLine;
                    result += "      Output : " + node.Output + Environment.NewLine;
                    result += "      Delta  : " + node.Delta + Environment.NewLine;
                    result += "      Bias   : " + node.Bias + Environment.NewLine;
                }
                foreach (var node in layer.Nodes)
                {
                    result += "  Connections : " + node.RightConnections.Count + Environment.NewLine;
                    result += "      From  : " + node.Code + Environment.NewLine;
                    foreach (var rightConnection in node.RightConnections)
                    {
                        result += "         To          : " + rightConnection.RightNode.Code + Environment.NewLine;
                        result += "         Weight      : " + rightConnection.Weight + Environment.NewLine;
                        result += "         WeightDelta : " + rightConnection.WeightDelta + Environment.NewLine;
                        result += "         PrevDelta   : " + rightConnection.PreviousWeightDelta + Environment.NewLine;
                    }
                }

            }
            return result;
        }

        public void Test(double[] input, double[] desired)
        {
            this.Run(input);
            this.LastTestingError = this.CalcError(this.Results(), desired);
            this.TestingErrorTotal += this.LastTestingError;
            this.TestingCount++;
        }

        public void SetLinearOutput()
        {
            this.LastLayer().SetTransferFunctions(TransferFunctions.Linear, TransferFunctions.LinearDerivitive);
        }
    }
}