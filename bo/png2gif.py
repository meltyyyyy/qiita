from PIL import Image

pictures = []
n_trials = 15

for i in range(n_trials):
    pic_name = 'bo_' + str(i) + '.png'
    img = Image.open(pic_name)
    pictures.append(img)

pictures[0].save('bo.gif', save_all=True, append_images=pictures[1:],
                 optimize=False, duration=500, loop=0)
