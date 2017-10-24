---
layout: post
title:  "Install Ruby on MacOS"
date:   2017-10-24 22:35:47 +0900
categories: how-to
comments: true
---

## Requirements

- [`homebrew`](https://brew.sh/)


## Install Ruby using *homebrew*

1. Install [`rbenv`](https://github.com/rbenv/rbenv) which is ruby version controller
```sh
$ brew install rbenv
```

2. Set up `rbenv` integration with shell
```sh
$ eval "$(rbenv init -)"
```

3. Install ruby version 2.3.0
```sh
# install ruby
$ rbenv install 2.3.0
# check installed version
$ rbenv versions
* system (set by /Users/ganbaatarbyambasuren/.rbenv/version)
  2.3.0
# activate ruby 2.3.0 version
$ rbenv global 2.3.0
# check version
$ ruby -v
ruby 2.3.0p0 (2015-12-25 revision 53290) [x86_64-darwin16]
```

## Reference
- https://brew.sh/
- https://github.com/rbenv/rbenv#homebrew-on-macos

{% include disqus.html %}
