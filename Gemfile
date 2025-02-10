source "https://rubygems.org"

# Use GitHub Pages instead of standalone Jekyll
gem "github-pages", group: :jekyll_plugins

# If you want to use a theme, choose one:
# gem "minima", "~> 2.5"  # Uncomment if using Minima
# gem "jekyll-theme-cayman", "~> 0.2.0"  # Uncomment if using Cayman (not needed if using `remote_theme`)

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
  gem "jekyll-include-cache"
  gem "jekyll-paginate"
  gem "jekyll-archives"
end

# Windows and JRuby platform dependencies
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem for JRuby
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# Other gems
gem "logger"
gem "csv"
gem "ostruct"
gem "base64"
