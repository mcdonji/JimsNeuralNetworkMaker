using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Emidium.Jnnm.Build
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("JNNM Build");
            string target = args[0];
            for (int i = 1; i < args.Length; i=i+2)
            {
                if (!(i + 1 >= args.Length))
                {
                    var buildSwitch = args[i];
                    var buildValue = args[i + 1];
                }
            }

        }
    }
}
