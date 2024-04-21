{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from nssvm_python.NSSVM.solver.NSSVM import NSSVM as OriginalNSSVM\n",
    "\n",
    "class NSSVMWithChecks(OriginalNSSVM):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "\n",
    "    def _my_cg(self, Q, y, E, b, cgtol, cgit, x):\n",
    "        # Call the original _my_cg method\n",
    "        result = super()._my_cg(Q, y, E, b, cgtol, cgit, x)\n",
    "        \n",
    "        # After the step, calculate the trust region activity and step size\n",
    "        # (Here you would implement the calculations based on the result and other parameters)\n",
    "        trust_region_active = ... # Your calculation here\n",
    "        step_size = ... # Your calculation here\n",
    "\n",
    "        # You can print the results or store them in the instance for later retrieval\n",
    "        self.trust_region_active = trust_region_active\n",
    "        self.step_size = step_size\n",
    "        print(f\"Trust region active: {trust_region_active}\")\n",
    "        print(f\"Step size: {step_size}\")\n",
    "\n",
    "        return result\n",
    "    \n",
    "model = NSSVMWithChecks()\n",
    "out = model.fit(X, y, pars)\n",
    "\n",
    "# After fitting, you can access the trust region activity and step size\n",
    "print(model.trust_region_active)\n",
    "print(model.step_size)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
