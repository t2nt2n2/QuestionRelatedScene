#import nltk
#from nltk.tokenize import sent_tokenize
import json
import csv
import random
#nltk.download('punkt')

def data_preprocessing(dir):
    with open(dir+'FriendsQA_desc.json') as data_file:    
        data = json.load(data_file)
        values = list(data.values())
        values.sort(key=lambda x:x["vid"])

        structures = []
        for i in range(len(values)):
            if values[i]['vid'].split('_')[-1]=="000":
                structure = {}
                structure['name'] = values[i]['vid'][:-4]
                structure['scene'] = values[i]
                structure['shot'] = []
                i=i+1
                while i<len(values) and values[i]['vid'].split('_')[-1]!="000":
                    structure['shot'].append(values[i])
                    i=i+1
                structures.append(structure)
        
        dataset = []
        for x in structures:
            data_json = {}
            data_json['name'] = x['name']
            data_json['subtitle'] = []
            with open(dir+'Subtitles/'+data_json['name'][:-3]+"subtitle.json") as data_file:    
                data = json.load(data_file)
                for subtitle_results in data["subtitle_results"]:
                    if data_json['name'] in [vid_name[:-4] for vid_name in subtitle_results['vid']]:
                        subtitle_results['script'].pop('st')
                        subtitle_results['script'].pop('et')
                        data_json['subtitle'].append(subtitle_results['script'])
            
            data_json['subtitle'] = ' '.join([temp['speaker'] + " : " + temp['utter'] for temp in data_json['subtitle']])
            
            data_json['shot_desc'] = []
            for y in x['shot']:
                for desc in y['desc']:
                    data_json['shot_desc'].append(desc)
                
            data_json['scene_desc'] = []
            for desc in x['scene']['desc']:
                pass
                #data_json['scene_desc'].extend(sent_tokenize(desc))
            

            data_json['question'] = []
            with open(dir+'FriendsQA.json') as data_file:
                 data = json.load(data_file)
                 for y in data:
                     if (y['vid'][:-4] ==  data_json['name']):
                         data_json['question'].append(y['que'])
            

            data_json['shot_desc'] = ' '.join(data_json['shot_desc'])
            dataset.append(data_json)
        return dataset


def writefile(dataset,dir,scene,question):
    f1 = open(dir, 'w', encoding='utf-8',newline='')
    wr1 = csv.writer(f1)
    cnt=0
    #scene = "s02e05"
    #question = "What happen at the time when  Ross and Chandler was in home ?"
    wr1.writerow(["index","shot_id","question",'subtitles','shot_descs','label'])
    for x in dataset:
        if x['name'].split('_')[0] == scene:
            wr = wr1
            wr.writerow([cnt,x['name'],question,x['subtitle'],x['shot_desc'],"1"])
            cnt+=1
    f1.close()
