config = {
    'extractor_batch_size': 32, 
    'model_name': 'mmt_lm', 
    'log_path': 'data/log',
    #'tokenizer': 'nltk',
    'tokenizer': 'nonword', # 'nltk' # need to check
    #'tokenizer': 'bert',
    'batch_sizes':  (8, 8, 8), # (16, 24, 12)
    'lower': True,
    'use_inputs':['que','answers','subtitle','speaker','images','sample_visual','filtered_visual','filtered_sub','filtered_speaker','filtered_image','que_len','ans_len','sub_len','filtered_visual_len','filtered_sub_len','filtered_image_len', 'filtered_person_full', 'filtered_person_full_len','per_person_features','text_masked','text_masked_l','token_type_ids','labels','labels_l','q_level_logic'],
    'stream_type': ['script',  'visual_bb', 'visual_meta'], #
    'cache_image_vectors': True,
    'image_path': 'data/AnotherMissOh/AnotherMissOh_images',
    'visual_path': 'data/AnotherMissOh/AnotherMissOh_Visual.json',
    'data_path': 'data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_script.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_script.json',
    'glove_path': "data/glove.6B.300d.txt",
    #'glove_path': "data/glove.840B.300d.txt", #"data/glove.6B.300d.txt", # "data/glove.6B.50d.txt"
    'vocab_path': "data/vocab.pickle",
    'val_type': 'all', # all | ch_only
    'pretrain_epochs': 2,
    'max_epochs': 50,
    'num_workers': 40, 
    #'image_dim': 512,  # hardcoded for ResNet18
    'image_dim': 2048, # hardcoded for ResNet152
    'n_dim': 300,  
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 1e-5, # pretrain: 1e-5, default: 1e-4
    'weight_decay': 1e-5,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'max_sentence_len': 30,
    'max_sub_len': 200,
    'max_text_len': 220,
    'max_image_len': 100,
    'shuffle': (True, False, False),
    'ckpt': 3
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    'ckpt'
    # 'feature_pooling_method', # useless
]
