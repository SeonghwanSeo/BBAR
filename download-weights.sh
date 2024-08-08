conda install --yes -c conda-forge gdown
mkdir -p ./test/pretrained_model
cd ./test/pretrained_model
gdown 'https://drive.google.com/uc?id=1t-gS7bCOxO5gqv8GXkluTzx1mO20_L0G'    # logp
gdown 'https://drive.google.com/uc?id=10UXskmoQQW6KZAeNSVqF4LliMZB9j5Iq'    # mw
gdown 'https://drive.google.com/uc?id=1897Gpa27jFOAYcGupdVGsPpMMbqDgcU9'    # qed
gdown 'https://drive.google.com/uc?id=1kRqaLIl152vw1yGzS2Dji3jPas9ksW0V'    # tpsa
gdown 'https://drive.google.com/uc?id=1udYJuz_p3pMxGBWsJyqHMcrT1l7saUkB'    # 3cl_affinity
gdown 'https://drive.google.com/uc?id=1lY5gU6YBDSZvqvEosEPlQ1w3HTe55sCJ'    # logp_tpsa
gdown 'https://drive.google.com/uc?id=1uFJbHMCRlXOTDIzWK-Dn6z8R9SeafDDx'    # mw_logp
cd ../../
