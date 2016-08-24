namespace Emidium.Jnnm.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class Layer
    {
        public string Id { get; set; }
        public int LayerIndex { get; set; }
        public bool IsInput { get; set; }
        public bool IsOutput { get; set; }
        public List<Node> Nodes = new List<Node>();

        public Layer(int layerIndex, bool isOutput, int numberOfNodes)
        {
            this.Id = Guid.NewGuid().ToString();
            this.LayerIndex = layerIndex;
            this.IsInput = layerIndex == 0;
            this.IsOutput = isOutput;
            for (int i = 0; i < numberOfNodes; i++)
            {
                var node = new Node(this, i);
//                if (isOutput)
//                {
//                    node.TransferFunction = TransferFunctions.Linear;
//                    node.TransferFunctionDerivitive = TransferFunctions.LinearDerivitive;
//                }
                this.Nodes.Add(node);
            }
        }

        public Layer LeftLayer { get; set; }
        public Layer RightLayer { get; set; }

        public void SetOutput(double[] inputs)
        {
            for (int i = 0; i < this.Nodes.Count; i++)
            {
                this.Nodes[i].Input = inputs[i];
                this.Nodes[i].Output = inputs[i];
            }
        }

        public void SetDelta(double[] desired)
        {
            for (int i = 0; i < this.Nodes.Count; i++)
            {
                this.Nodes[i].SetDeltaFromDesired(desired[i]);
            }
        }

        public void SetBias(double trainingrate, double momentum)
        {
            foreach (Node node in this.Nodes)
            {
                node.SetBias(trainingrate, momentum);
            }
        }

        public double AverageDelta()
        {
            return this.Nodes.Sum(n => n.Delta) / this.Nodes.Count;
        }

        public void ForwardPropagate()
        {
            this.Nodes.ForEach(node => node.ForwardPropagate());
        }

        public void SetWeights(double trainingRate, double momentum)
        {
            foreach (var node in this.Nodes)
            {
                node.RightConnections.ForEach(rcon => rcon.SetWeight(trainingRate, momentum));
            }
        }

        public void BackPropagateDeltas()
        {
            this.Nodes.ForEach(node => node.BackPropagateDelta());
        }

        public void SetTransferFunctions(Func<double, double> tf, Func<double, double> tfd)
        {
            this.Nodes.ForEach(n => { n.TransferFunction = tf; n.TransferFunctionDerivitive = tfd; });
        }
    }
}