from . import RedditInfo
import praw
def LoginReddit():
    reddit = praw.Reddit(client_id=RedditInfo.REDDIT_CLIENT_ID(),
                        client_secret=RedditInfo.REDDIT_SECRET(),
                        user_agent=RedditInfo.USER_AGENT())
    return reddit
