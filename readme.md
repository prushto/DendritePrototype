DendritePrototype is a simplified software emulation of a retrieval accelerator that retrieves documents encoded into a sparse, binary-weight bag-of-words (BoW) representation.

The accelerator operates in three phases: (1) memorization, during which each document’s BoW vector is sequentially programmed into the chip’s memory, (2) search, during which a query’s BoW vector is presented to the chip to score the memories, and (3) ranking, during which the highest scoring memories stream out of the accelerator in decreasing relevance. To perform these operations, the accelerator consists of three modules: 

<ol>
<li>a <b>scoring unit</b> for each document in the corpus,</li>
<li>a programmable <b>routing network</b> that connects a subset of the |V | input wires onto each scoring unit, and</li>
<li>a <b>network</b> that collects and compares the output of scoring units and streams out their index in order of decreasing score.</li>
</ol>


Each of these three modules may be customized. See unit_interfaces.py for details. 

Edit config.json to effect your changes. 

Execute the emulation using `python DendritePrototype.py`.

See requirements.txt for packages.

Other config options:
<ul>
<li>labels_filepath: the emulation will evaluate its recall against the supplied labels and output the results to the given filepath</li>
<li>output_filepath: text file output for the evaluation results</li>
<li>rank_num: R hyperparameter for recall evaluation (default=1000)</li>
</ul>