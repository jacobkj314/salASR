import argparse
import csv
from pathlib import Path 
from os.path import exists 
import os
import regex as re
import glob
from evaluate import load
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

ALL_AUDIO_RS = [2, 5, 8]
ALL_AUDIO_STRATEGIES = ["t", "r", "b"]

R_2_LABEL = {
    2: 0.2,
    5: 0.5,
    8: 0.8
}

STRATEGY_2_LABEL = {
    "t": "top",
    "r": "random",
    "b": "bottom",
}

werMetric = load("wer", cache_dir="/scratch/general/vast/u1419542/huggingface_cache")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-userTranscriptions",
    type=str,
    help="Path to csv file containing user transcriptions",
)

parser.add_argument(
    "-originalTranscriptions",
    type=str,
    help="Path to directory containing files with user transcriptions",
)
#------------------------------------------------------------
def checkIfExists(path:str, isDir:bool=False, createIfNotExists:bool=False, errTrace:str="checkIfExists") -> None: 
    if isDir and not path.endswith("/"):
        raise ValueError(f"[{errTrace}] Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"[{errTrace}] {path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"[{errTrace}] {path} is not a file!")
#------------------------------------------------------------
def checkFile(fileName:str, fileExtension:str=None, errTrace:str="checkFile") -> None:
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[{errTrace}] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[{errTrace}] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[{errTrace}] {fileName} is not a file!")
#------------------------------------------------------------
def readFile(filePath:str, errTrace:str="readFile") -> list:
    if filePath.endswith(".csv"):
        with open(filePath, "r") as f:
            data = list(csv.DictReader(f))
    elif filePath.endswith(".txt"):
        with open(filePath, "r") as f:
            data = list(f.readlines())
    else: 
        raise ValueError("[{}] Unsupported file type: {}".format(errTrace, filePath.split(".")[-1]))
    return data
#------------------------------------------------------------
def getFiles(dirPath:str, fileExtension:str=".txt", errTrace:str="getFiles") -> list:
    filePattern = os.path.join(dirPath,  "*{}".format(fileExtension))
    testFiles = glob.glob(filePattern)
    try: 
        re.compile(filePattern)
    except: 
        raise RuntimeError(f"[{errTrace}] {filePattern} is not a valid regular expression!")
    if len(testFiles) == 0:
        raise RuntimeError(f"[{errTrace}] {filePattern} did not match any file in {dirPath}!")
    return testFiles
#------------------------------------------------------------
def getTranscriptions(transcriptionFiles:list, errTrace:str="getTranscriptions") -> dict:
    transcriptions = {}
    for tFile in transcriptionFiles:
        audioID = tFile.split("_")[0].split("/")[-1]
        assert audioID.isdigit()
        audioID = int(audioID)
        if audioID in transcriptions.keys():
            raise RuntimeError("[{}] More than one transcription file with the ID: {}!".format(errTrace, audioID))
        transcriptions[audioID] = readFile(tFile)[0].strip()
    return transcriptions
#------------------------------------------------------------
def main(errTrace="main"):
    args = parser.parse_args()
    checkFile(args.userTranscriptions, ".csv")
    checkIfExists(args.originalTranscriptions, isDir=True, createIfNotExists=False)

    userTranscriptions = readFile(args.userTranscriptions)
    print("Loaded user transcriptions.")

    transcriptionFiles = getFiles(args.originalTranscriptions, ".txt")
    transcriptions = getTranscriptions(transcriptionFiles)
    print("Loaded original transcriptions.")

    wers = {}
    for strategy in ALL_AUDIO_STRATEGIES:
        wers[strategy] = {} 
        for r in ALL_AUDIO_RS:
            wers[strategy][r] = []

    for uTrans in tqdm(userTranscriptions, desc="userTranscriptions"):
        audioInd2ID = {}
        for k in uTrans.keys():
            if k.startswith("Input.audio"):
                audioInd = k[len("Input.audio"):] 
                assert audioInd.isdigit()
                audioInd = int(audioInd)
                audioID = (uTrans[k].split("/")[-1]).split("_")[0]
                assert audioID.isdigit()
                audioID = int(audioID)
                audioR = (uTrans[k].split("/")[-1]).split("_")[1]
                assert audioR.isdigit()
                audioR = int(audioR)
                assert audioR in ALL_AUDIO_RS
                audioStrategy = (uTrans[k].split("/")[-1]).split("_")[2].split(".")[0]
                assert audioStrategy in ALL_AUDIO_STRATEGIES
                audioInd2ID[audioInd] = {
                    "id": audioID,
                    "r": audioR, 
                    "strategy": audioStrategy, 
                }
        audioIDpos = [v["id"] for v in audioInd2ID.values()]
        predictions = [""]*len(audioIDpos)
        references = [""]*len(audioIDpos)
        predictionsR = [""]*len(audioIDpos)
        predictionsStrategy = [""]*len(audioIDpos)
        for k in uTrans.keys():
            if k.startswith("Answer.transcription"):
                audioInd = k[len("Answer.transcription"):] 
                assert audioInd.isdigit()
                audioInd = int(audioInd)
                if audioInd not in audioInd2ID.keys():
                    raise RuntimeError("[{}] No input audio but transcription found for input audio with index {}!".format(errTrace, audioInd))
                audioID = audioInd2ID[audioInd]["id"]
                predictions[audioIDpos.index(audioID)] = uTrans[k]
                predictionsR[audioIDpos.index(audioID)] = audioInd2ID[audioInd]["r"]
                predictionsStrategy[audioIDpos.index(audioID)] = audioInd2ID[audioInd]["strategy"]
                if audioID not in transcriptions.keys():
                    raise RuntimeError("[{}] Could not find original transcription for audio sample {}!".format(errTrace, audioID))
                references[audioIDpos.index(audioID)] = transcriptions[audioID]
        for ref, pred, r, strategy in zip(references, predictions, predictionsR, predictionsStrategy):
            # print("Reference: {}, Transcription: {}".format(ref.lower(), pred.lower()))
            wers[strategy][r].append(
                werMetric.compute(
                    references=[ref.lower()], 
                    predictions=[pred.lower()]
                )
            )

    checkIfExists("./plots/", isDir=True, createIfNotExists=True)
    plt.clf()
    plt.title("WER for user transcriptions")
    plt.xlabel("Masking percentage (r)")
    plt.ylabel("Average Word Error Rate (WER)")
    pltLegend = []
    print("Average WERs by strategy and masking percentage")
    for strategy in wers:
        print("{}:".format(STRATEGY_2_LABEL[strategy]))
        Rs = []
        scores = []
        for r in wers[strategy]:
            print("\t{}: {}".format(R_2_LABEL[r], np.mean(wers[strategy][r])))
            Rs.append(R_2_LABEL[r])
            scores.append(np.mean(wers[strategy][r]))
        plt.plot(Rs, scores)
        pltLegend.append(STRATEGY_2_LABEL[strategy])
    plt.legend(pltLegend)
    plt.savefig("./plots/userStudy.png")
#------------------------------------------------------------
if __name__=="__main__":
    main()