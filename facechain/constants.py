neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, slim body, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality'

base_models = [
    {'name': 'leosamsMoonfilm_filmGrain20',
    'model_id': 'ly261666/cv_portrait_model',
    'revision': 'v2.0',
    'sub_path': "film/film",
    'style_list': ['工作服(Working suit)', '盔甲风(Armor)','T恤衫(T-shirt)','汉服风(Hanfu)','女士晚礼服(Gown)','赛博朋克(Cybernetics punk)','凤冠霞帔(Chinese traditional gorgeous suit)']},
    {'name': 'MajicmixRealistic_v6',
    'model_id': 'YorickHe/majicmixRealistic_v6',
    'revision': 'v1.0.0',
    'sub_path': "realistic",
     # zju_02 add here
    'style_list': ['枪林弹雨风(CallOfDuty)','印度风(India)', '海洋风(ocean)', '花园风(flowers)','武侠风(kongfu)']},
    #,'拍立得风(Polaroid style)', '仙女风(Fairy style)', '古风(traditional chinese style)', '壮族服装风(Zhuang style)', '欧式田野风(European fields)', '自然户外风(natural outdoor)','武林（gongfu）'
]

styles = [
    # zju_02 add here
    # 使命召唤7.1， very nice
    {'name': '枪林弹雨风(CallOfDuty)',   # 名称，要与前面style_list对应
     'img': './style_image/CallOfDuty.jpg',    # 在web界面上的展示图片，自行添加
     'model_id': 'Tyytyy528/ShortSemester2023_zju_02',    # modelscope上的模型id
     'revision': 'v2.0.9',    # 版本号，千万不能错！
     'bin_file': 'CallOfDuty7.safetensors',    # LoRA模型文件
     'multiplier_style': 0.27,
     'multiplier_human': 0.95,
     'add_prompt_style': 'solo, rifle, realistic, manly, gloves, military uniform, goggles overhead, bulletproof vest, holding weapon,headset, fire, explosion, buildings, street'},    # 提示词，重要
    {'name': '印度风(India)',
     'img': './style_image/india.jpg',
     'model_id': 'lljjcc/IndianSarres',
     'revision':'v1.0.0',
     'bin_file': 'Indian saree.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'add_prompt_style': 'wear indian style costume, gold ornament, gorgeous background, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality, indian dance, indian street, street photography'},
    {'name': '海洋风(ocean)',
     'img': './style_image/DreamyOcean.png',
     'model_id': 'MushroomLyn/artist',
     'revision': 'v2.0.0',
     'bin_file': 'mine.safetensors',
     'multiplier_style': 0.5,
     'multiplier_human': 0.95,
     'add_prompt_style': 'delicate, gentle eye, ocean, white shirt, sunlit, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality'},
     {'name': '花园风(flowers)',
     'img': './style_image/outdoor.jpg',
     'model_id': 'lljjcc/outdoor',
     'revision': 'v1.0.0',
     'bin_file': 'outdoor photo_v2.0.safetensors',
     'multiplier_style': 0.86,
     'multiplier_human': 0.95,
     'add_prompt_style': 'girl,flower, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality'},
    {'name': '武侠风(kongfu)',
     'img': './style_image/wulin.jpg',
     'model_id': 'lljjcc/gongfu',
     'revision': 'v1.0.0',
     'bin_file': '武侠飘逸高清脸_出男脸必备_男女脸都漂亮_v1.0.safetensors',
     'multiplier_style': 0.86,
     'multiplier_human': 0.95,
     'add_prompt_style': '1boy, martial arts costume,  forest, long hair, period costume, martial arts, solo, jewelry, long hair, earrings, necklace, shut up, black eyes, belt, tassel, upper_body, original photo, masterpiece, high detail face, slim body, realistic, best quality, chinoiserie, ksword, ancient architectural background'}
    # zju_02 end here
]

pose_models = [
    {'name': '无姿态控制(No pose control)'},
    {'name': 'pose-v1.1-with-depth'},
    {'name': 'pose-v1.1'}
]

pose_examples = {
    'man': [
        ['./poses/man/pose1.png'],
        ['./poses/man/pose2.png'],
        ['./poses/man/pose3.png'],
        ['./poses/man/pose4.png']
    ],
    'woman': [
        ['./poses/woman/pose1.png'],
        ['./poses/woman/pose2.png'],
        ['./poses/woman/pose3.png'],
        ['./poses/woman/pose4.png'],
    ]
}
