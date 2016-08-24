namespace Emidium.Jnnm.Core
{
    using System.Collections.Generic;
    using System.Linq;

    public class NeuralNetworkSpecification
    {
        public readonly List<int> LayerNodes = new List<int>();
        public int NumberOfLayers => this.LayerNodes.Count;

        public int NumberOfInputs => this.LayerNodes.First();

        public int NumberOfOutputs => this.LayerNodes.Last();

        public NeuralNetworkSpecification(List<int> layerNodes)
        {
            this.LayerNodes = layerNodes;
        }

        public NeuralNetworkSpecification(int inputCount, int layerCount, int nodeCount, int outputCount)
        {
            this.LayerNodes.Add(inputCount);
            for (int layerNum = 0; layerNum < layerCount; layerNum++)
            {
                this.LayerNodes.Add(nodeCount);
            }
            this.LayerNodes.Add(outputCount);
        }

        public bool IsOutput(int layerIndex)
        {
            return layerIndex == this.NumberOfInputs;
        }

        public override string ToString()
        {
            return "NN (" + string.Join(" ", this.LayerNodes) + ")";
        }
    }
}