namespace Emidium.Jnnm.Core
{
    using System;

    public class TransferFunctions
    {
        public static Func<double, double> None = input => 0; 
        public static Func<double, double> NoneDerivitive = input => 0;
        public static Func<double, double> Sigmoid = input => 1 /( 1 + (Math.Exp((-1 * input))));
        public static Func<double, double> SigmoidDerivitive = input => (Sigmoid(input) * (1 - Sigmoid(input)));
        public static Func<double, double> Linear = input => input;
        public static Func<double, double> LinearDerivitive = input => 1;
        public static Func<double, double> Gaussian = input => Math.Exp(-1 * Math.Pow(input,2));
        public static Func<double, double> GaussianDerivitive = input => (-2 * input * Gaussian(input));
        public static Func<double, double> RationalSigmoidLike = input => (input / (1 + Math.Sqrt(1 + (input * input))));
        public static Func<double, double> RationalSigmoidLikeDerivitive = input => { var val =  Math.Sqrt(1 + (input * input)); return (1 / (val * (1 + val)));};
    }
}