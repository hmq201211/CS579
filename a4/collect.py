"""
Collect data.
"""
import json
import sys
import time
import requests
from TwitterAPI import TwitterAPI

# api keys
consumer_key = 'BTVZVgFqcEEK5K3iMcAPVitQk'
consumer_secret = 'vSiUMGxfEhi713qidb20c2Fy0xKBH2Md5LCWYsfDU1HgtY9yPZ'
access_token = '1059495093760073735-FDOSPNB0Fxij1pEYehbBtuG6WiSqsV'
access_token_secret = '43pMQ4uJwE9SbqJX1PnCSZH3VeN2npdhQQMR7iBuQ3JBG'


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    with open(filename) as openHandler:
        result = [s.split()[0] for s in openHandler.readlines()]
    openHandler.close()
    return result


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [users['id'] for u in users]
    [6253282, 783214]
    """

    response = robust_request(twitter, "users/lookup", {"screen_name": screen_names})
    return response


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    response = robust_request(twitter, "friends/ids", {'screen_name': screen_name, 'count': 5000}).json()['ids']
    return sorted(response)


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    for user in users:
        user['friends'] = get_friends(twitter, user['screen_name'])


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    sorted(users, key=lambda x: x['screen_name'])
    for user in users:
        print(user['screen_name'], len(user['friends']))


def get_census_names():
    """ Fetch a list of common male/female names from the census.
    For ambiguous names, we select the more frequent gender."""
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                      for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                        for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                      males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                        females_pct[f] > males_pct[f]])
    return male_names, female_names


def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()


def sample_tweets(twitter, limit, male_names, female_names):
    tweets, labels = [], []
    while True:
        try:
            # Restrict to U.S.
            for response in twitter.request('statuses/filter',
                                            {'locations': '-124.637,24.548,-66.993,48.9974'}):
                if 'user' in response:
                    name = get_first_name(response)
                    if name in male_names:
                        tweets.append(response)
                        labels.append(0)
                        if len(tweets) % 100 == 0:
                            print('found %d tweets' % len(tweets))
                        if len(tweets) >= limit:
                            return tweets, labels
                    elif name in female_names:
                        tweets.append(response)
                        labels.append(1)
                        if len(tweets) % 100 == 0:
                            print('found %d tweets' % len(tweets))
                        if len(tweets) >= limit:
                            return tweets, labels
        except:
            print("Unexpected error:", sys.exc_info()[0])


def save_users(users, path):
    for u in users:
        json.dump(u, open(path + u['screen_name'] + ".txt", 'w'))


def save_tweets(tweets, labels, path):
    json.dump(tweets, open(path + 'raw_tweets.txt', 'w'))
    json.dump(labels, open(path + 'labels.txt', 'w'))


def main():
    print("collect file works")
    cluster_path = './shared/cluster_data/'
    classifer_path = './shared/classifer_data/'
    print('Starting to build Twitter connection.')
    twitter = get_twitter()
    print('Twitter connection built')
    candidates_file_names = './shared/candidates.txt'
    print('Starting to read screen_names_file: %s' % candidates_file_names)
    screen_names = read_screen_names(candidates_file_names)
    print('reading task finished')
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    print("Begin to collect friends's ids")
    add_all_friends(twitter, users)
    print("collecting friends's ids task finished")
    print("Begin to write cluster data to file")
    save_users(users, cluster_path)
    print("writing cluster data to file task finished")
    print("begin to collect test names for male and female")
    male_names, female_names = get_census_names()
    print('found ' + str(len(male_names)) + ' male names and ' + str(len(female_names)) + ' female names')
    print("begin to collect tweets for classifer")
    raw_tweets, labels = sample_tweets(twitter, 5000, male_names, female_names)
    print("collected " + str(len(raw_tweets)) + " tweets")
    print("collected " + str(len(labels)) + " labels")
    print("begin to write tweets and labels to file")
    save_tweets(raw_tweets, labels, classifer_path)
    print("writing classifer data to file task finished")
    print("collect file finished")


if __name__ == "__main__":
    main()
