---
title: First Steps in R
authors: [jan]
summary: Learn Data Science with R
categories:
- course
tags:
#- course
#- tidyverse
#- rmarkdown
- R
- DataScience
date: "2019-08-03T00:00:00Z"

# Optional external URL for project (replaces project detail page).
#external_link: "https://ohsu-math630.netlify.com/"

image:
  caption: '[Photo by Lee Campbell on Unsplash](https://unsplash.com/photos/6njoEbtarec)'
  focal_point: Smart

links:
#- icon: door-open
#  icon_pack: fas
#  name: website
#  url: https://ohsu-conj620.netlify.com/
- icon: github
  icon_pack: fab
  name: materials
  url: https://github.com/kirenz/first-steps-in-r
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

# First Steps in R


![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/R_logo.svg/200px-R_logo.svg.png)

Download the PDF [R overview](https://github.com/kirenz/first-steps-in-r/blob/master/R_overview.pdf) to get an overview about R and a list of helpful resources (you need to download the file in order to use the embedded links).

## Installing R

The first step is to install R. You can download and install R from the [Comprehensive R Archive Network](https://cran.r-project.org/) (CRAN).

Windows:
- Open the [Comprehensive R Archive Network](https://cran.r-project.org/).
- Click on “CRAN”. You’ll see a list of mirror sites, organized by country.
- Select a site near you.
- Click on “Windows” under “Download and Install R”.
- Click on “base”.
- Click on the link for downloading the latest version of R (an .exe file).
- When the download completes, double-click on the .exe file and answer the usual questions.

Mac:
- Open the [Comprehensive R Archive Network](https://cran.r-project.org/).
- Click on “CRAN”.
- You’ll see a list of mirror sites, organized by country.
- Select a site near you.
- Click on “MacOS X”.
- Click on the .pkg file for the latest version of R, under “Files:”, to download it.
- When the download completes, double-click on the .pkg file and answer the usual questions.


## Installing RStudio

The next step is to install **RStudio**, a free and open-source integrated development environment (IDE) for R. You can use it for viewing and running R scripts.

- Go to [RStudio Download](https://www.rstudio.com/products/rstudio/#Desktop)
- Click the Download RStudio Desktop button.
- Select the installation file for your system.
- Run the installation file.


## Learn R Basics

First of all, you can take an online course to master the basics of R: Visit the interactive [R-Course](https://www.datacamp.com/getting-started?step=2&track=r) from DataCamp. With the knowledge gained in this courses, you will be ready to undertake your first very own data analysis.

There are also open and free resources and reference guides for R. Two examples are:

- [Quick-R](http://www.statmethods.net/): a quick online reference for data input, basic statistics and plots
- [R reference card (PDF)](https://cran.r-project.org/doc/contrib/Short-refcard.pdf) by Tom Short

Two key things you need to know about R is that you can get help for a function using `help` or `?`, like this:

```{r,eval=FALSE}
?install.packages
help("install.packages")
```

and the hash character represents comments, so text following these
characters is not interpreted:

```{r}
##This is just a comment
```

## Installing R Packages

The first R command we will run is `install.packages`.

An R package is a collection of functions, data, and documentation that extends the capabilities of base R.
Many of these functions are stored in CRAN. You can easily install packages from within RStudio if you know
the name of the packages.

As an example, we are going to install the
package `dplyr` which we use in our first data
analysis examples:

```{r,eval=FALSE}
install.packages("dplyr")
```

We can then load the package into our R sessions using the `library` function:

```{r}
library(dplyr)
```

From now on you will see that we sometimes load packages without
installing them. This is because you only need to install a package once,
but you need to reload it with the command `library` every time you start
a new R session.

If you try to load a package and get an error, it probably means you need to install it first.

Review the [dplyr-documentation](https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html) to get an overview about the different functionalities of this package.
