neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, slim body, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality'

base_models = [
    {'name': 'leosamsMoonfilm_filmGrain20',
    'model_id': 'ly261666/cv_portrait_model',
    'revision': 'v2.0',
    'sub_path': "film/film",
    'style_list': ['默认风格(default style)', '凤冠霞帔(Chinese traditional gorgeous suit)']},
    {'name': 'MajicmixRealistic_v6',
    'model_id': 'YorickHe/majicmixRealistic_v6',
    'revision': 'v1.0.0',
    'sub_path': "realistic",
    'style_list': ['未知(unknown)', '瑶(yaoyao)', '洛丽塔(loli)', '户外婚纱(outdoor)', '冬季汉服(Chinese winter hanfu)', '校服风(School uniform)', '婚纱风(Wedding dress)', '拍立得风(Polaroid style)', '仙女风(Fairy style)']},
]

cloth_prompt = [
    {'name': '工作服(working suit)', 'prompt': 'wearing high-class business/working suit, simple background, high-class pure color background'},  # male and female
    {'name': '盔甲风(armor)', 'prompt': 'wearing silver armor, simple background, high-class pure color background'},  # male
    {'name': 'T恤衫(T-shirt)', 'prompt': 'wearing T-shirt, simple background, high-class pure color background'},  # male and female
    {'name': '汉服风(hanfu)', 'prompt': 'wearing beautiful traditional hanfu, upper_body, simple background, high-class pure color background'},  # female
    {'name': '女士晚礼服(gown)', 'prompt': 'wearing an elegant evening gown, simple background, high-class pure color background'},  # female
    {'name': '赛博朋克(cybernetics punk)', 'prompt': 'white hair, neon glowing glasses, cybernetics, punks, robotic, AI, NFT art, Fluorescent color, ustration'}, # male and female
]

styles = [
    {'name': '默认风格(default style)'},
    {'name': '凤冠霞帔(Chinese traditional gorgeous suit)',
     'model_id': 'ly261666/civitai_xiapei_lora',
     'revision': 'v1.0.0',
     'bin_file': 'xiapei.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '汉服风(hanfu)',
     'add_prompt_style': 'red, hanfu, tiara, crown'},
    {'name': '未知(unknown)',
     'model_id': '/mnt/workspace/facechain/LORA',
     'revision': 'v1.0.0',
     'bin_file': 'fashigirl-v5.5-lora-naivae-64dim.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '未知(unknown)',
     'add_prompt_style': 'woman, face, doll'},
    {'name': '瑶(yaoyao)',
     'model_id': '/mnt/workspace/facechain/LORA',
     'revision': 'v1.0.0',
     'bin_file': 'yaoyao_v0.4.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '瑶(yaoyao)',
     'add_prompt_style': 'masterpiece, best quality,solo,blush,full body, 1gir,yaoyao,silver hair,dress,pointy ears,blue capelet,white horns,bow,sitting, looking at viewer, white simple background, <lora:yaoyao_v0.4:1>'},
    {'name': '洛丽塔(loli)',
     'model_id': '/mnt/workspace/facechain/LORA',
     'revision': 'v1.0.0',
     'bin_file': 'lo_dress_gothic_style2_v2.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '洛丽塔(loli)',
     'add_prompt_style': 'realistic, photorealistic, masterpiece, best quality, 1girl, long black straight hair, solo, sitting,in castle, night, looking at viewer, lolita_dress, black pantyhose,long dress'},
    {'name': '户外婚纱(outdoor)',
     'model_id': '/mnt/workspace/facechain/LORA',
     'revision': 'v1.0.0',
     'bin_file': 'wedding.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '户外婚纱(outdoor)',
     'add_prompt_style': 'bride wearing a white wedding dress,simple and elegant style'},
     {'name': '冬季汉服(Chinese winter hanfu)',
     'model_id': 'YorickHe/Winter_hanfu_lora',
     'revision': 'v1.0.0',
     'bin_file': 'Winter_Hanfu.safetensors',
     'multiplier_style': 0.3,
     'multiplier_human': 0.95,
     'cloth_name': '冬季汉服(winter hanfu)',
     'add_prompt_style': 'red hanfu, winter hanfu, cloak, photography, warm light, sunlight, majestic snow scene,  close-up, front view, soft falling snowflakes, jewelry, enchanted winter wonderland'},
     {'name': '校服风(School uniform)',
     'model_id': 'YorickHe/JK_uniform_lora',
     'revision': 'v1.0.0',
     'bin_file': 'jk_uniform.safetensors',
     'multiplier_style': 0.2,
     'multiplier_human': 0.95,
     'cloth_name': '校服风(school uniform)',
     'add_prompt_style': 'JK_style, white short-sleeved JK_shirt, dark blue JK_skirt, bow JK_tie,  close-up, looking at viewer, night, Tokyo street, night city scape'},
     {'name': '婚纱风(Wedding dress)',
     'model_id': 'YorickHe/outdoor_photo_lora',
     'revision': 'v1.0.0',
     'bin_file': 'outdoor_photo_v2.0.safetensors',
     'multiplier_style': 0.3,
     'multiplier_human': 0.95,
     'cloth_name': '户外婚纱照(Outdoor wedding photo)',
     'add_prompt_style': 'white wedding dress, 1girl, summer, flower, happy atmosphere, sunlight, Waist Shot,'},
     {'name': '拍立得风(Polaroid style)',
     'model_id': 'YorickHe/polaroid_lora',
     'revision': 'v1.0.0',
     'bin_file': 'InstantPhotoX3.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '拍立得风(Polaroid style)',
     'add_prompt_style': '1girl, close-up, face shot, stylish outfit, fitted jeans, oversized jacket, fashionable accessories, cityscape backdrop, rooftop or high-rise balcony, dynamic composition, engaging pose, soft yet striking lighting, shallow depth of field, bokeh from city lights, naturally blurred background'},
     {'name': '仙女风(Fairy style)',
     'model_id': 'YorickHe/fairy_lora',
     'revision': 'v1.0.0',
     'bin_file': 'fairy.safetensors',
     'multiplier_style': 0.25,
     'multiplier_human': 0.95,
     'cloth_name': '仙女风(Fairy style)',
     'add_prompt_style': 'a beautiful fairy standing in the middle of a flower field, petals, close-up, warm light, light green atmosphere, white atmosphere, in the style of celebrity photography, soft, romantic scenes, flowing fabrics, light white and light orange, high resolution'},
]

inpaint_default_positive = 'beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
inpaint_default_negative = 'hair, teeth, sketch, duplicate, ugly, huge eyes, text, logo, worst face, strange mouth, nsfw, NSFW, low quality, worst quality, worst quality, low quality, normal quality, lowres, watermark, lowres, monochrome, naked, nude, nsfw, bad anatomy, bad hands, normal quality, grayscale, mural,'

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

