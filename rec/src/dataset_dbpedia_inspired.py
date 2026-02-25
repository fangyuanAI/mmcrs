import json
import os

import torch
from loguru import logger
import json
import os
import tqdm
import torch
from loguru import logger
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import pickle
class DBpedia:
    def __init__(self, dataset, debug=False):
        self.debug = debug

        self.dataset_dir = os.path.join('rec_data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self):
        edge_list = set()  
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(self.relation2id)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
        }
        return kg_info
    

class Co_occurrence:
    def __init__(self, dataset,split, entity_max_length, all_items,n_entity,debug=False):
        self.debug = debug
        self.entity_max_length =entity_max_length
        self.all_items = set(all_items)
        input_file = 'rec_data/inspired/edge_index_c.pt'
        self.edge_index_c = torch.load(input_file)

    def get_entity_co_info(self):
        co_info = {
            'edge_index_c': self.edge_index_c,
        }
        return co_info




class text_sim:
    def __init__(self, pad_entity_id, num_entities):
        dataset_dir = 'rec_data/inspired'
        data_file = os.path.join(dataset_dir, 'id_embeddings_text.json')
        self.pad_entity_id = pad_entity_id
        self.num_entities = num_entities
        self.prepare_data(data_file)

    def prepare_data(self, data_file):         
        with open(data_file, 'r', encoding='utf-8') as f:
            id_embeddings_raw = json.load(f)
            
        self.embeddings = torch.zeros(self.num_entities, 768)
        for ent_id_str, emb in id_embeddings_raw.items():
            ent_id = int(ent_id_str)
            if ent_id < self.num_entities:
                self.embeddings[ent_id] = torch.tensor(emb)
        
        # Đặc trưng cho pad_token và các thực thể thiếu
        self.embeddings[self.pad_entity_id] = torch.ones(768)

    def get_entity_ts_info(self):
        return {
            'text_embeds': self.embeddings
        }


class image_sim:
    def __init__(self, pad_entity_id, num_entities):
        dataset_dir = 'rec_data/inspired'
        data_file = os.path.join(dataset_dir, 'id_embeddings_image.json')
        self.pad_entity_id = pad_entity_id
        self.num_entities = num_entities
        self.prepare_data(data_file)

    def prepare_data(self, data_file):        
        with open(data_file, 'r', encoding='utf-8') as f:
            id_embeddings_raw = json.load(f)
            
        self.embeddings = torch.zeros(self.num_entities, 768)
        for ent_id_str, emb in id_embeddings_raw.items():
            ent_id = int(ent_id_str)
            if ent_id < self.num_entities:
                self.embeddings[ent_id] = torch.tensor(emb)
        
        self.embeddings[self.pad_entity_id] = torch.ones(768)

    def get_entity_is_info(self):
        return {
            'image_embeds': self.embeddings
        }