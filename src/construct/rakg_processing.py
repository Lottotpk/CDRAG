import json
import ast
import pandas as pd
import uuid

def str_to_dict():
    for i in range(1, 106):
        with open(f"./data/RAKG/{i}.json", "r") as f:
            data = json.load(f)
        data = ast.literal_eval(data)
        
        with open(f"./data/RAKG/{i}.json", "w") as f:
            json.dump(data, f, indent=2)


def find_id_text(txt_map, name, ent):
    txt_list = None
    for _, entity in ent.items():
        if name == entity["name"]:
            txt_list = entity["description"].split(";;;")
    res = []
    if txt_list is not None:
        for txt in txt_list:
            res.append(txt_map[txt])
    return res


def find_id_ent(ent_map, txt, ent):
    res = []
    for _, entity in ent.items():
        desc_list = entity["description"].split(";;;")
        if txt in desc_list and entity["name"] in ent_map:
            res.append(ent_map[entity["name"]])
    return res


def to_parquet():
    
    # Store list for entities, relationships, and text units, respectively.
    ent_id, ent_human_readable_id, title, type, ent_description, ent_text_unit_ids, frequency = [], [], [], [], [], [], []
    rel_id, rel_human_readable_id, source, target, rel_description, weight, rel_text_unit_ids = [], [], [], [], [], [], []
    txt_id, txt_human_readable_id, text, n_tokens, entity_ids, relationships_ids = [], [], [], [], [], []
    
    for i in range(1, 106):
        
        # 1. Read files
        with open(f"./data/RAKG/{i}.json", "r") as f:
            data = json.load(f)
        with open(f"./data/RAKG/{i}_entities.json", "r") as f:
            ent = json.load(f)
        with open(f"./data/RAKG/{i}_ner.json", "r") as f:
            ner = json.load(f)

        # 2. create entities-id, relationship-id, and text_units-id mapping
        ent_map = {}
        rel_map = {}
        txt_map = {}
        for entity in data["entities"]:
            ent_map[entity["name"]] = uuid.uuid4().hex
        for relation in data["relations"]:
            rel_map[tuple(relation)] = uuid.uuid4().hex
        for _, entity in ner.items():
            txt_map[entity["description"]] = uuid.uuid4().hex
        
        # 3. construct entities dataframe
        count = 1
        for entity in data["entities"]:
            ref = find_id_text(txt_map, entity["name"], ent)

            ent_id.append(ent_map[entity["name"]])
            ent_human_readable_id.append(count)
            title.append(entity["name"])
            type.append(entity["type"])
            ent_description.append("\n".join([f"{key}: {value}" for key, value in entity["attributes"].items()]))
            ent_text_unit_ids.append(ref)
            frequency.append(len(ref))
            count += 1
        
        # 4. construct relationships dataframe
        count = 1
        for relation in data["relations"]:
            rel_id.append(rel_map[tuple(relation)])
            rel_human_readable_id.append(count)
            source.append(relation[0])
            target.append(relation[2])
            rel_description.append(relation[1])
            weight.append(1.0)
            rel_text_unit_ids.append(None) 
            count += 1
        
        # 5. construct text units dataframe
        count = 1
        for _, entity in ner.items():
            ref = find_id_ent(ent_map, entity["description"], ent)

            txt_id.append(txt_map[entity["description"]])
            txt_human_readable_id.append(count)
            text.append(entity["description"])
            n_tokens.append(None)
            entity_ids.append(ref)
            relationships_ids.append(None)
            count += 1

    # 6. Convert into data frame and into parquet
    pd.DataFrame(data={
        "ent_id": ent_id, 
        "ent_human_readable_id": ent_human_readable_id, 
        "title": title, 
        "type": type, 
        "ent_description": ent_description, 
        "ent_text_unit_ids": ent_text_unit_ids, 
        "frequency": frequency
    }).to_parquet("./data/output/entities.parquet", index=False)
    pd.DataFrame(data={
        "rel_id": rel_id, 
        "rel_human_readable_id": rel_human_readable_id, 
        "source": source, 
        "target": target, 
        "rel_description": rel_description, 
        "weight": weight, 
        "rel_text_unit_ids": rel_text_unit_ids
    }).to_parquet("./data/output/relationships.parquet", index=False)
    pd.DataFrame(data={
        "txt_id": txt_id, 
        "txt_human_readable_id": txt_human_readable_id, 
        "text": text, 
        "n_tokens": n_tokens, 
        "entity_ids": entity_ids, 
        "relationships_ids": relationships_ids 
    }).to_parquet("./data/output/text_units.parquet", index=False)


if __name__ == "__main__":
    str_to_dict()
    # to_parquet()