namespace Emidium.Jnnm.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class Node
    {
        public string Id { get; set; }

        public List<NodeConnection> LeftConnections = new List<NodeConnection>();

        public List<NodeConnection> RightConnections = new List<NodeConnection>();
        public string Code => this.Layer.LayerIndex + "-" + this.NodeIndex;
        public int NodeIndex { get; set; }
        public Layer Layer { get; set; }
        public double Bias { get; set; }
        public double PreviousBiasDelta { get; set; }
        public double Input { get; set; }
        public double Output { get; set; }
        public double Delta { get; set; }
        public Func<double, double> TransferFunction { get; set; }
        public Func<double, double> TransferFunctionDerivitive { get; set; }

        public Node(Layer layer, int nodeIndex)
        {
            this.NodeIndex = nodeIndex;
            this.Layer = layer;
            this.TransferFunction = TransferFunctions.Sigmoid;
            this.TransferFunctionDerivitive = TransferFunctions.SigmoidDerivitive;
            this.Bias = 0;
            this.PreviousBiasDelta = 0;
            this.Delta = 0;
            this.Input = 0;
            this.Output = 0;
        }

        public void SetDeltaFromDesired(double desired)
        {
            this.SetDelta(this.Output - desired);
        }

        public void SetDelta(double difference)
        {
            this.Delta = difference * this.TransferFunctionDerivitive(this.Input);
        }

        public void ForwardPropagate()
        {
            var input = this.LeftConnections.Sum(connection => connection.WeightedOutput);
            this.Input = input + this.Bias;
            this.Output = this.TransferFunction(this.Input);
        }

        public void BackPropagateDelta()
        {
            var weightedRightHandDeltas = this.RightConnections.Sum(rcon => (rcon.Weight * rcon.RightNode.Delta));
            this.Delta = weightedRightHandDeltas * this.TransferFunctionDerivitive(this.Input);
        }

        public void SetBias(double trainingrate, double momentum)
        {
            double biasDelta = trainingrate * this.Delta;
            this.Bias = this.Bias - biasDelta + momentum * this.PreviousBiasDelta;
            this.PreviousBiasDelta = biasDelta;
        }
    }
}