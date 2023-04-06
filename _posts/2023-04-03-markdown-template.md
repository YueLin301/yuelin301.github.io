---
title: Markdown Syntax
date: 2022-04-03 15:48:44 +0800
categories: [Notes]
tags: [note]     # TAG names should always be lowercase
math: True
---

> Adapted from [this post](https://chirpy.cotes.page/posts/text-and-typography/)
{: .prompt-info }

---
## Prompts
> green tip
{: .prompt-tip }

> blue info
{: .prompt-info }

> yellow warning
{: .prompt-warning }

> red danger
{: .prompt-danger }

```
> green tip
{: .prompt-tip }

> blue info
{: .prompt-info }

> yellow warning
{: .prompt-warning }

> red danger
{: .prompt-danger }
```

---

## Titles
```
# H1
## H2
### H3
```

---

## List
### Ordered List
```
1. 111
2. 222
3. 333
```

### Unordered List
The indentation determines the structure, and the choice of `*` or `-` or `+` is insignificant.

```
- 111
    - 222
        - 333
```

### TODO List
The indentation determines the structure, and the choice of `*` or `-` or `+` is insignificant.

```
- [ ] Job
    - [x] Step 1
    - [x] Step 2
    - [ ] Step 3
```

---

## Description
```
Sun
: the star around which the earth orbits
```

---

## Emphasis
```
*Italian*
**Bold**
```

---

## Hyper Link
```
[name](url)
```

---

## Block Quote
```
> blahblah
```

---

## Table

| Company                      | Contact          | Country |
| --- | --- | --- |
| Alfreds Futterkiste          | Maria Anders     | Germany |
| Island Trading               | Helen Bennett    | UK      |
| Magazzini Alimentari Riuniti | Giovanni Rovelli | Italy   |

---

## Footnote
Click the hook will locate the footnote[^footnote], and here is another footnote[^fn-nth-2].

[^footnote]: The footnote source
[^fn-nth-2]: The 2nd footnote source

---

## Filepath
```
`/path/to/the/file.extend`{: .filepath}.
```

## Mathematics
(front matter: math: True)

$$ \sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6} $$
```
$$ \sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6} $$
```

## Image & Video

### Default (with caption)
![pic name](/assets/img/meow.png){: width="300" height="300" }
_caption_

### Left Aligned
![pic name](/assets/img/meow.png){: width="300" height="300" .w-75 .normal}

### Float to Left
![pic name](/assets/img/meow.png){: width="150" height="150" .w-50 .left}
Praesent maximus aliquam sapien. Sed vel neque in dolor pulvinar auctor. Maecenas pharetra, sem sit amet interdum posuere, tellus lacus eleifend magna, ac lobortis felis ipsum id sapien. Proin ornare rutrum metus, ac convallis diam volutpat sit amet. Phasellus volutpat, elit sit amet tincidunt mollis, felis mi scelerisque mauris, ut facilisis leo magna accumsan sapien. In rutrum vehicula nisl eget tempor. Nullam maximus ullamcorper libero non maximus. Integer ultricies velit id convallis varius. Praesent eu nisl eu urna finibus ultrices id nec ex. Mauris ac mattis quam. Fusce aliquam est nec sapien bibendum, vitae malesuada ligula condimentum.

### Float to Right
![pic name](/assets/img/meow.png){: width="150" height="150" .w-50 .right}
Praesent maximus aliquam sapien. Sed vel neque in dolor pulvinar auctor. Maecenas pharetra, sem sit amet interdum posuere, tellus lacus eleifend magna, ac lobortis felis ipsum id sapien. Proin ornare rutrum metus, ac convallis diam volutpat sit amet. Phasellus volutpat, elit sit amet tincidunt mollis, felis mi scelerisque mauris, ut facilisis leo magna accumsan sapien. In rutrum vehicula nisl eget tempor. Nullam maximus ullamcorper libero non maximus. Integer ultricies velit id convallis varius. Praesent eu nisl eu urna finibus ultrices id nec ex. Mauris ac mattis quam. Fusce aliquam est nec sapien bibendum, vitae malesuada ligula condimentum.

### Video
<!-- {% include embed/youtube.html id='Balreaj8Yqs' %} -->
![video name](/assets/my_paper/RHex_T3/workspace2.mp4)