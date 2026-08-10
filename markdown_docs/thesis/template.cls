% Classe supsistudent
% versione 0.02, 27 settembre 2012
% Autore: Amos Brocco <amos.brocco@supsi.ch>
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}[2012/09/27 v0.02 SUPSI Student,
amos.brocco@supsi.ch]

% Modifiche impaginazione
% Autori: Mikol Faro e Sandra Gabaglio <sandra.gabaglio@supsi.ch>
\usepackage{url}
\usepackage{multicol}
% Package di base
\LoadClass[11pt]{book}
\usepackage[english]{babel}
%\usepackage[paper=a4paper,top=4.0cm,bottom=2cm,left=3cm,right=2cm]{geometry}
\usepackage[paper=a4paper,top=4.0cm,bottom=3cm]{geometry}
\usepackage[utf8]{inputenc}
%\usepackage{slantsc}
\usepackage{array}
\usepackage{amsmath}
\usepackage{titlesec}
% Font (Helvetica, simile a Arial)
\usepackage[T1]{fontenc}
\usepackage[scaled]{helvet}
% Spaziatura
\usepackage{setspace}
% Referenze
\usepackage{varioref}
% Figure
\usepackage{graphicx}
% Bibliografia
%\usepackage{natbib}
% Per adattare margini
\usepackage[strict]{changepage}
% Footer logo
\usepackage{eso-pic}
% Numero pagine
\usepackage{lastpage}
\assignpagestyle{\chapter}{fancy}

\usepackage{amssymb}
\usepackage{algorithm} % https://ctan.org/pkg/algorithms
\usepackage{algpseudocodex}

% Comandi
\makeatletter
	\newcommand{\titolo}[1]{\def \@title {#1}}
	\newcommand{\studente}[1]{\def \@student {#1}}
	\newcommand{\relatore}[1]{\def \@relator {#1}}
	\newcommand{\correlatore}[1]{\def \@correlator {#1}}
	\newcommand{\committente}[1]{\def \@client {#1}}
	\newcommand{\corso}[1]{\def \@course {#1}}
	\newcommand{\modulo}[1]{\def \@module {#1}}
	\newcommand{\anno}[1]{\def \@year {#1}}
	\studente{}
	\relatore{}
	\correlatore{}
	\committente{}
	\corso{}
	\modulo{}
	\anno{} 
\makeatother

\newcommand{\linea}[0]{\vspace*{2em}\noindent\makebox[\linewidth][l]{\rule{\textwidth}{1pt}}\hspace*{\fill}\\}

% Formattazione predefinita
\renewcommand{\rmdefault}{ptm} % Times
\renewcommand{\sfdefault}{phv} % Helvetica
\renewcommand{\ttdefault}{cmtt}
\renewcommand{\familydefault}{\sfdefault} % Font standard per il documento
%\titleformat{\chapter}{\Huge}{\thesection}{18pt}{}
%\titleformat{\section}{\large\bfseries}{\thesubsection}{11pt}{}
%\titleformat{\subsection}{\large\bfseries}{\thesubsubsection}{10pt}{}
%\titleformat{\paragraph}{\large\bfseries}{\paragraph}{9pt}{}
%\renewcommand{\normalsize}{\fontsize{11pt}{11pt}\fontseries{m}\selectfont}
\newcommand{\etichettetitolo}{\fontsize{8pt}{8pt}\fontseries{m}\selectfont}
\newcommand{\nomititolo}{\vspace{1em}\fontsize{14pt}{14pt}\fontseries{bx}\selectfont}
\setlength\parindent{0pt}
\setlength{\headsep}{40pt}
\setlength{\headheight}{17pt}


\makeatletter
\newcommand{\campo}[1]{\ifx {#1} \@empty
					--
				\else
					\nomititolo{{#1}}
				\fi}
\makeatother				

%\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\space#1}{}} 


% Intestazione e pié di pagina
\makeatletter
	\usepackage{fancyhdr}
	\pagestyle{fancy}
	\renewcommand{\headrulewidth}{1pt}
	\renewcommand{\footrulewidth}{1pt}
	\fancyhf{}
	\fancyhead[LO]{
	  \includegraphics[width=70mm]{images/template/logoheader.png}
	}
	\fancyhead[RE]{ 
		\leftmark
	}
	\setlength{\footskip}{50pt}
	\fancyfoot[LE]{
	    \small{\@title} \AddToShipoutPicture*{\put(58,28){\includegraphics[height=44mm]{images/template/logoverticale.png}}}
                %\small{\@title} \AddToShipoutPicture*{\put(58,154){\scalebox{-1}{\includegraphics[height=44mm]{logoverticale}}}}
	}
	\fancyfoot[RO]{
	  %\small{\@title} \AddToShipoutPicture*{\put(527,154){\scalebox{-1}[-1]{\includegraphics[height=44mm]{logoverticale}}}}
	      \small{\@title} \AddToShipoutPicture*{\put(527,28){{\includegraphics[height=44mm]{images/template/logoverticale.png}}}}
	}
	\fancyhead[LE, RO]{
	 % \small{\thepage/\pageref{LastPage}}
		 \small{\thepage}
	}
	%\fancyhead[RO]{
	 % \small{\thepage/\pageref{LastPage}}
	%}
\makeatother

\renewcommand{\chaptermark}[1]{ \markboth{#1}{} }
% Pagina titolo
\makeatletter
\def \maketitle {%
	\begin{titlepage}
	% Allinea a sinistra
	\raggedright
	% Intestazione SUPSI e logo
	\vspace{20mm}
	{\parindent-1mm\makebox(0,0)[lb]{\includegraphics[width=170mm]{images/template/logoprima.png}}}
	\\
	{\Huge \textrm{\@title}} \\
	\vspace{10mm}
	\linea
		\begin{minipage}[t]{0.49\textwidth}
			\begin{flushleft}
				\etichettetitolo{Student/s} \\
				\campo{\@student}
			\end{flushleft}
		\end{minipage}
		\hspace*{\fill}
		\begin{minipage}[t]{0.49\textwidth}
			\begin{flushleft}
				\etichettetitolo{Supervisor} \\
				\campo{\@relator} \\
				\linea
				\vspace*{0.3em} % Fix spaziatura
				\etichettetitolo{Co-supervisor} \\
				\campo{\@correlator} \\
				\linea  
				\vspace*{0.3em} % Fix spaziatura
				\etichettetitolo{Host} \\
				\campo{\@client} \\	
			\end{flushleft}
		\end{minipage} \\
		\vspace{5mm}		
		\linea
		\begin{minipage}[t]{0.49\textwidth}
			\begin{flushleft}
				\etichettetitolo{Course} \\
				\campo{\@course} \\
			\end{flushleft}
		\end{minipage}
		\hspace*{\fill}
		\begin{minipage}[t]{0.49\textwidth}
			\begin{flushleft}
				\etichettetitolo{Module} \\
				\campo{\@module} \\
			\end{flushleft}
		\end{minipage} \\
		\vspace*{1em} 
		\linea
		\vspace*{0.3em}
		\etichettetitolo{Year} \\
		\campo{\@year} \\
 		\vfill

 		\begin{minipage}[b]{0.49\textwidth}
 		\begin{flushleft}
 		\linea
 		\vspace*{0.3em} % Fix spaziatura
	 	\etichettetitolo{Date} \\
	 	\campo{\@date} \\
 		\end{flushleft}
 		\end{minipage}
 		\hspace*{\fill}
 		
 		% \begin{minipage}[b]{0.49\textwidth}
 		% \begin{flushright}
 		\leavevmode\makebox(0,0){\put(434,180){\includegraphics[height=71mm]{images/template/logoverticale.png}}}
 		% \end{flushright}
 			
 		% \end{minipage}
		
		%Aggiunge una pagina bianca senza header e footer dietro la copertina
		\newpage
		\thispagestyle{empty}
		\mbox{}
		\newpage
	\end{titlepage}
	\setlength{\voffset}{-15mm}
	\singlespacing
}
\makeatother