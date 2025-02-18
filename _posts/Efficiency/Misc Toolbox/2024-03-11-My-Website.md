---
title: My Website
date: 2024-03-11 00:20:00 +0800
categories: [Efficiency, Misc Toolbox]
tags: [Tech, Toolbox]
---


## Jekyll

- 读作"街口"
- [【转载 - Jekyll - 静态网站生成器教程双语字幕】](https://www.bilibili.com/video/BV1qs41157ZZ/?share_source=copy_web&vd_source=b3cf9eb7cfe43c730613c5158a38e978)


## 迁移和部署

### 迁移

比如要换电脑，那么这个博客怎么重新装起来
1. 复制一份这个文件
2. 安装环境
	1. `ruby` and `jekyll`: https://jekyllrb.com/docs/installation/macos/
	2. `bundle install`
	3. if something is wrong, remove the `Gemfile.lock` and follow the prompted suggestions in the terminal

### 部署

这样就默认是`127.0.0.1:4000`，然后只能本地访问
```
bundle exec jekyll serve --livereload
```

如果要让局域网里的其他设备都能访问，那就：
```
bundle exec jekyll serve --livereload --host 0.0.0.0
```

- 然后去终端输`ifconfig`查看本机地址是啥，一般是en0，然后我的情况是`192.168.2.182`，那手机上输入`http://192.168.2.182:4000`就能访问了
- 更简单的方法是直接去settings里搜network然后看wifi下面的IP address就行了


## Github Pages

### Custom Domain

避坑：你得有顶级域名apex domain才能用，必须是.com后缀的。腾讯云上的这个后缀域名基本是一年85块。其他8块一年的域名拿来不能在Github Pages这配

步骤：
1. 腾讯云，实名注册，挑域名，买域名；几个步骤中间要审核
2. 然后管理域名的解析
   1. 在Github右上角点开头像的那个setting里面去设置，点page，点增加，然后他会给你TXT的记录名和记录值
   2. 去腾讯云，管理解析，输入
3. 去Github Pages项目里的设置，把域名输进去然后验证

好，到了这一步了我才发现只支持.com后缀域名，哈哈，白干

### CDN

**Q:** github pages的服务器是不是在海外

**A:** GitHub Pages的服务器主要位于海外。GitHub作为一个全球性的平台，其数据中心分布在世界各地，但主要集中在美国。这意味着对于大多数国家和地区的用户来说，访问GitHub Pages托管的网站时，数据可能需要跨越国际网络，这可能会影响到访问速度和稳定性。不过，GitHub也采用了一些技术和策略来优化全球用户的访问体验，例如通过CDN（内容分发网络）来加速内容的加载。

CDN可以将网站的内容缓存到全球各地的节点上，这样无论是在中国大陆还是美国，用户都可以从最近的节点获取数据，大大提高了访问速度和稳定性。常用的CDN服务提供商包括Cloudflare、Akamai、阿里云CDN和腾讯云CDN等。

**Q:** 所以说当我github pages项目更新之后，是不是过几天等CDN缓存到节点上了，那么其他地区的人打开速度就会变快了?

**A:** 是的，当你更新了GitHub Pages项目后，这些更新需要一定的时间来传播到CDN（内容分发网络）的各个节点上。一旦更新的内容被缓存到距离用户更近的节点，那么用户访问该内容的速度就会变快。这个过程通常发生得很快，但确切的时间可能会因为不同的CDN提供商和网络条件而有所不同。

GitHub Pages使用的CDN可以帮助加速全球的内容分发，包括静态文件如HTML、CSS和JavaScript等。这意味着无论用户位于哪里，都可以从最近的CDN节点获取数据，从而减少了加载时间。

需要注意的是，CDN缓存更新的速度也取决于CDN配置中设置的缓存过期时间（TTL，Time To Live）。如果设置的TTL较长，更新后的内容在所有CDN节点上完全刷新并可供所有用户访问可能需要更多时间。相反，如果TTL较短，内容更新会更快地传播到各个节点，但这可能会增加服务器的负载和流量成本。