<h1>
    salASR: Salience for Automatic Speech Recognition
</h1>
<p>
    This repository contains pre for the project on extending common explainability approaches used in Deep Learning literature to the task of Automatic Speech Recognition. This project was done for the course CS 6966 Local Explanations for Deep Learning Models taught by Professor Ana Marasovic. 
</p>
<h2>
    Contributors
</h2>
<ul>
    <li>
        Jacob Johnson
    </li>
    <li>
        Gurunath Parasaram
    </li>
    <li>
        <a href="https://rishanthrajendhran.github.io/" target="_blank">Rishanth Rajendhran</a>
    </li>
</ul>
<h2>
    Dataset
</h2>
<p>
    We use the <a href="https://huggingface.co/datasets/librispeech_asr" target="_blank">librispeech_asr</a> dataset for this project.
    <h3>
        Data Format
    </h3>
    <p>
        <pre>
{
    'chapter_id': int,
    'file': str,
    'audio': {
        'path': str,
        'array': array(float32),
        'sampling_rate': int
    },
    'id': str,
    'speaker_id': int,
    'text': str
}
        </pre>
    </p>
</p>
<h2>
    Getting started
</h2>
<ul>
    <li>
        Clone the github repo:<br/>
        <pre>
git clone https://github.com/jacobkj314/salASR
        </pre>
    </li>
    <li>
        Install conda if you don't have it already
        <pre>
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash Miniconda3-py37_4.12.0-Linux-x86_64.sh
        </pre>
    </li>
    <li>
        Once you have installed conda, use it to create and set up the conda environment
        <pre>
conda create -n salASR python=3.8
conda activate salASR
pip install -r requirements.txt
        </pre>
    </li>
    <li>
        If you wish to finetune whisper, create and set up another conda environment
        <pre>
conda create -n finetuneWhisperEnv python=3.9
conda activate finetuneWhisperEnv
pip install -r requirementsFinetuneWhisper.txt
        </pre>
    </li>
</ul>