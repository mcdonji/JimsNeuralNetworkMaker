namespace Emidium.Jnnm.Core
{
    using System;

    public class NodeConnection
    {
        public NodeConnection(Node thisLayerNode, Node nextLayerNode)
        {
            this.Id = Guid.NewGuid().ToString();
            this.LeftNode = thisLayerNode;
            this.RightNode = nextLayerNode;
            this.Weight = 0;
            this.PreviousWeightDelta = 0;
        }

        public string Id { get; set; }
        public Node LeftNode { get; set; }
        public Node RightNode { get; set; }

        public double Weight { get; set; }
        public double WeightDelta { get; set; }
        public double PreviousWeightDelta { get; set; }

        public double WeightedOutput => this.Weight * this.LeftNode.Output;

        public void SetWeight(double trainingrate, double momentum)
        {
            var weightDelta = (trainingrate * this.RightNode.Delta * this.LeftNode.Output) + (momentum * this.PreviousWeightDelta) ;
            this.Weight = this.Weight - weightDelta;
            this.PreviousWeightDelta = weightDelta;
        }
    }
}