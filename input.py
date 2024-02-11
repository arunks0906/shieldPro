from fuzzywuzzy import fuzz

def inputs(request):
    user_id = request.POST['username']
    profile_pic = request.POST['profile'] == 'on'
    fullname = request.POST['fullname']
    description = request.POST['description']
    n_post = request.POST['post']
    n_followers = request.POST['followers']
    n_followings = request.POST['followings']
    private = request.POST['account_status'] == 'on'
    url = request.POST['url'] == 'on'

    num_user = 0
    for i in user_id:
        if i.isnumeric():
            num_user += 1

    num_name = 0
    for i in fullname:
        if i.isnumeric():
            num_name += 1

    ratio = fuzz.ratio(user_id.lower(), fullname.lower())
    if ratio > 50:
        similarity = 1
    else:
        similarity = 0

    if len(fullname) == 0:
        flag = 0
    else:
        flag = num_name/len(fullname)

    sample = {'profile pic': profile_pic,
              'nums/length username': num_user/len(user_id),
              'fullname words': len(fullname.split(' ')),
              'nums/length fullname': flag,
              'name==username': similarity,
              'description length': len(description),
              'external URL': url,
              'private': private,
              '#posts': n_post,
              '#followers': n_followers,
              '#follows': n_followings
              }

    return sample
